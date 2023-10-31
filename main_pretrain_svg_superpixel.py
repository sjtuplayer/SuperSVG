import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import random
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import timm
from models.wgan import Wgan
# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from models.attn_painter_superpixel import AttnPainterSVG
from PIL import Image
from engine_pretrain_superpixel import train_one_epoch
from skimage.segmentation import slic
import numpy as np
from torchvision.utils import save_image
width=224
class AttrImgDataset(Dataset):
    def __init__(self,data_root='/home/huteng/dataset/imagenet/val',bs=24,with_lable=True):
        self.bs=bs
        self.data_root=data_root
        if with_lable is True:
            subdirs=[os.path.join(self.data_root,i) for i in os.listdir(data_root)]
            self.paths=[]
            for subdir in subdirs:
                self.paths+=[os.path.join(subdir,i) for i in os.listdir(subdir)]
            #self.paths=[os.path.join(self.data_root,i) for i in os.listdir(subdirs)]
        else:
            self.paths=[os.path.join(self.data_root,i) for i in os.listdir(data_root)]
        self.l=len(self.paths)
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((width, width)),
            transforms.ToTensor(),
            ])
        self.resize_128=transforms.Resize([128,128])
        self.resize=transforms.Resize([width,width])
    def __getitem__(self, index):
        index=random.randint(0,len(self.paths)-1)
        image = Image.open(self.paths[index%self.l]).convert('RGB')
        image=np.array(image)/255.0
        segments = torch.from_numpy(slic(image, n_segments=16, sigma=5, compactness=30))
        image=torch.from_numpy(image).permute(2,0,1).float()
        segment_num=segments.max()
        mask_idx=random.randint(1,segment_num)
        seg_mask = (segments == mask_idx).int()
        idxs = torch.nonzero(seg_mask)
        x1, x2, y1, y2 = idxs[:, 0].min(), idxs[:, 0].max(), idxs[:, 1].min(), idxs[:, 1].max()
        mask = seg_mask[x1:x2 + 1, y1:y2 + 1].unsqueeze(0)
        mask=self.resize(mask)
        image=self.resize(image)
        mask = (mask > 0.5).int().float()
        return image,mask
    def __len__(self):
        return min(self.l,100000)
        #return max(self.l,6000)


def get_args_parser():
    parser = argparse.ArgumentParser('Encoder training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--type', default='attn', type=str,
                        help='model types', choices=['conv', 'attn', 'attnv2'])

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=6, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/home/huteng/dataset/imagenet/train', type=str,
                        help='dataset path')
    parser.add_argument('--label_name', default='')
    parser.add_argument('--output_dir', default='./output_superpixel',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--control_num', action='store_true')
    parser.add_argument('--report_time', action='store_true')
    parser.add_argument('--mask_loss', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    args.log_dir=args.output_dir
    misc.init_distributed_mode(args)
    if args.wandb and args.rank==0:
        import wandb
        wandb.init(config=args,
                   project='SuperSVG',
                   name='supersvg',
                   reinit=True)
    else:
        wandb=None
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation

    #dataset_train = AttrImgDataset('/home/huteng/dataset/test-imgs')
    dataset_train = AttrImgDataset(args.data_path)
    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # define the model
    model=AttnPainterSVG(stroke_num=128,path_num=4,width=width,control_num=args.control_num)
    #ckpt = torch.load('/home/huteng/SVG/AttnPainter/output-test/checkpoints-best.pt')
    # ckpt = torch.load('/home/huteng/SVG/AttnPainter/output-64/checkpoints-imagenet.pt')
    ckpt=torch.load('/home/huteng/SVG/AttnPainter/output_superpixel/checkpoints128_paths-mask_loss.pt')
    # print(ckpt.keys())
    # new_ckpt={}
    # for key in ckpt.keys():
    #     if 'feature_extractor' in key:
    #         new_ckpt[key]=ckpt[key]
    model.load_state_dict(ckpt,strict=True)

    critic=None
    # checkpoint_model = torch.load('renderer-oil-FCN.pkl', map_location='cpu')
    #
    # msg = model.render.load_state_dict(checkpoint_model, strict=False)
    model.to(device)

    model_without_ddp = model

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    print(args.distributed,'hhhhhhhhhhhh')
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp.encoder, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args,
            epoch_id=epoch,
            wandb=wandb
        )
        # if args.output_dir and (epoch % 2 == 0 or epoch + 1 == args.epochs):
        if args.distributed:
            if args.rank==0:
                torch.save(model.module.state_dict(),args.output_dir+'/checkpoints%s.pt'%args.label_name)
        else:
            torch.save(model.state_dict(), args.output_dir + '/checkpoints%s.pt' % args.label_name)
        if critic is not None:
            torch.save(critic.net.state_dict(), args.output_dir + '/wgan-checkpoints%s.pt'%args.label_name)
        # misc.save_model(
        #     args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
        #     loss_scaler=loss_scaler, epoch=0)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
