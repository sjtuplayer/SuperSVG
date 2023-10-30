import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import timm

# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from models.attn_painter import AttnPainterSVG

from PIL import Image
from engine_pretrain import train_one_epoch

width=128
class AttrImgDataset(Dataset):
    def __init__(self,data_root='/home/huteng/dataset/imagenet/val'):
        self.data_root=data_root
        self.paths=[os.path.join(self.data_root,i) for i in os.listdir(data_root)]
        self.l=len(os.listdir(data_root))
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((width, width)),
            transforms.ToTensor(),
            ])
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    def __getitem__(self, index):
        img = Image.open(self.paths[index%self.l]).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return max(self.l,6000)


def get_args_parser():
    parser = argparse.ArgumentParser('Encoder training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--block_num', default=1, type=int,
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
    parser.add_argument('--data_path', default='/home/huteng/dataset/celeba-hq', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
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
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--name', type=str, default='art', )
    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # model=AttnPainterSVG(stroke_num=128,path_num=4,width=width)
    # ckpt=torch.load('/home/huteng/SVG/AttnPainter/output_autoregression/checkpoints-1.pt')
    # model.load_state_dict(ckpt)

    from models.attn_painter import AttnPainterSVG
    model = AttnPainterSVG(stroke_num=128, path_num=4, width=width,control_num=False)
    ckpt = torch.load('/home/huteng/SVG/AttnPainter/output-test/checkpoints-best.pt')
    model.load_state_dict(ckpt)

    # checkpoint_model = torch.load('renderer-oil-FCN.pkl', map_location='cpu')
    #
    # msg = model.render.load_state_dict(checkpoint_model, strict=False)
    model.to(device)

    model_without_ddp = model

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256


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
    name='art'
    dir='/home/huteng/SVG/AttnPainter/test-data/%s'%name
    files=[os.path.join(dir,i) for i in os.listdir(dir)]
    model.eval()
    l2_loss=torch.nn.MSELoss()
    from torchvision.utils import save_image
    average_loss=0
    block=args.block_num
    os.makedirs('output/block=%d/%s' % (block, name),exist_ok=True)
    print(len(files))
    for idx,file in enumerate(files):
        if block==1:
            loader = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ])
            img = Image.open(file).convert('RGB')
            img = loader(img).unsqueeze(0).cuda()
            model.width=512
            with torch.cuda.amp.autocast():
                # output, gt = model(img,save_svg_path='output/block=%d/%s/%d.svg' % (block,name,idx))
                output, current_prediction = model(img,canvas=torch.zeros_like(img).to(img.device),step=0,num=64)
            output=output[:, :3, :, :]
            average_loss+=float(l2_loss(output,img))
            # save_image(torch.cat([gt, output], dim=0), 'tmp-backup.jpg',
            #            normalize=False)
            # exit()
            # save_image(torch.cat([gt, output], dim=0), 'output/block=%d/%s/%d.jpg' % (block,name,idx), normalize=False)
        else:
            # loader = transforms.Compose([
            #     transforms.Resize((width*block, width*block)),
            #     transforms.ToTensor(),
            # ])
            # ori_img = Image.open(file).convert('RGB')
            # ori_img = loader(ori_img).unsqueeze(0).cuda()
            # img=ori_img.resize(1,3,block,width,block,width)
            # img=img.permute(0,2,4,1,3,5)
            # print(img.shape)
            # img=img.resize(block*block,3,width,width)
            # with torch.cuda.amp.autocast():
            #     output, gt = model(img)
            # output=output[:, :3, :, :]
            # average_loss+=float(l2_loss(output,gt))
            # output=output.resize(1,block,block,3,width,width)
            # output=output.permute(0,3,1,4,2,5)
            # output=output.resize(1,3,width*block,width*block)
            # save_image(torch.cat([ori_img,output],dim=0),'output/%d.jpg'%idx,normalize=False)
            loader = transforms.Compose([
                transforms.Resize((width * block, width * block)),
                transforms.ToTensor(),
            ])
            ori_img = Image.open(file).convert('RGB')
            ori_img = loader(ori_img).unsqueeze(0).cuda()
            img = ori_img.resize(1, 3, block, width, block, width)
            img = img.permute(0, 2, 4, 1, 3, 5)
            print(img.shape)
            img = img.resize(block * block, 3, width, width)
            #save_image(img, 'output/%d-0.jpg' % idx, normalize=False)
            with torch.cuda.amp.autocast():
                strokes = model.predict_path(img)
            print(strokes.shape)
            for i in range(block):
                for j in range(block):
                    block_id=i*block+j
                    strokes[block_id,:,:-3:2]=(1/block)*j+strokes[block_id,:,:-3:2]/block
                    strokes[block_id,:,1:-3:2] = (1 / block) * i + strokes[block_id,:,1:-3:2] / block
            strokes=strokes.resize(1,block*block*128,27)
            model.width=width*block
            output=model.rendering(strokes)
            output = output[:, :3, :, :]
            average_loss += float(l2_loss(output, ori_img))
            # output = output.resize(1, block, block, 3, width, width)
            # output = output.permute(0, 3, 1, 4, 2, 5)
            # output = output.resize(1, 3, width * block, width * block)
            save_image(output, 'output/block=%d/%s/%s' % (block, name,file.split('/')[-1]),
                       normalize=False)
            # save_image(torch.cat([ori_img, output], dim=0), 'output/block=%d/%s/%d.png' % (block,name,idx), normalize=False)
        print(average_loss/(idx+1))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
