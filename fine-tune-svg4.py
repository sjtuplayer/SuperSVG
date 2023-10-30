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
import time
# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from models.attn_painter import AttnPainterSVG
from PIL import Image
from engine_pretrain import train_one_epoch
#靠id_map筛选不可见map,并靠优化延缓边界效应
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

    model=AttnPainterSVG(stroke_num=128,path_num=4,width=width)
    ckpt=torch.load('/home/huteng/SVG/AttnPainter/output-test/checkpoints.pt')
    model.load_state_dict(ckpt)

    # checkpoint_model = torch.load('renderer-oil-FCN.pkl', map_location='cpu')
    #
    # msg = model.render.load_state_dict(checkpoint_model, strict=False)
    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

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

    dir='/home/huteng/SVG/AttnPainter/test-data/art'
    files=[os.path.join(dir,i) for i in os.listdir(dir)]
    model.eval()
    l2_loss=torch.nn.MSELoss()
    from torchvision.utils import save_image
    average_loss=0
    block=5
    for idx,file in enumerate(files):
        if idx!=15:
            continue
        loader = transforms.Compose([
            transforms.Resize((width * block, width * block)),
            transforms.ToTensor(),
        ])
        ori_img = Image.open(file).convert('RGB')
        ori_img = loader(ori_img).unsqueeze(0).cuda()
        img = ori_img.resize(1, 3, block, width, block, width)
        img = img.permute(0, 2, 4, 1, 3, 5)
        img = img.resize(block * block, 3, width, width)
        #save_image(img, 'output/%d-0.jpg' % idx, normalize=False)
        with torch.cuda.amp.autocast():
            strokes = model.predict_path(img)
        # for i in range(block*block):
        #     for k in range(128):
        #         strokes[i,k,-3:]=k
        # output = model.rendering(strokes)
        # for i in output.flatten():
        #     print(i)
        for i in range(block):
            for j in range(block):
                block_id=i*block+j
                strokes[block_id,:,:-3:2]=(1/block)*j+strokes[block_id,:,:-3:2]/block
                strokes[block_id,:,1:-3:2] = (1 / block) * i + strokes[block_id,:,1:-3:2] / block
        strokes=strokes.resize(1,block*block*128,27).detach().float()
        model.width=width*block
        id_strokes=strokes.clone()
        new_strokes=strokes.clone()[0]
        # output = model.rendering(new_strokes.unsqueeze(0), save_svg_path='output/test0.svg')
        # save_image(output[:, :3, :, :], 'output/test0.jpg')
        for i in range(id_strokes.size(1)):
            id_strokes[0,i,-3:]=i
        id_map=model.rendering(id_strokes)
        id_map = id_map[0, 0, :, :]
        threshhold=0.05
        ids=[]
        for i in range(1,block):
            line1=i*width-1
            line2=i*width
            mapping={}
            flag=torch.zeros(int(id_map.max()))
            for j in range(block*width):
                id1=int(id_map[line1][j])
                id2=int(id_map[line2][j])
                if id1!=id_map[line1][j] or id2!=id_map[line2][j]:
                    continue
                if id1 not in ids:
                    ids.append(id1)
                if id2 not in ids:
                    ids.append(id2)
        print('stage1 done!')
        strokes=strokes.detach()
        optimized_strokes=[]
        # for id in ids:
        #     strokes[0:1, id, :].requires_grad=True
        #     optimized_strokes.append(strokes[0:1,id,:])
        new_strokes=[]
        for i in range(strokes.size(1)):
            new_strokes.append(strokes[0,i,:])

        for i in range(len(new_strokes)):
            new_strokes[i].requires_grad = True
            if i in ids:
                optimized_strokes.append(new_strokes[i])
        optimizer = torch.optim.AdamW(new_strokes, lr=0.01, betas=(0.9, 0.95))
        # strokes.requires_grad=True
        #optimizer = torch.optim.AdamW([strokes], lr=0.001, betas=(0.9, 0.95))
        s=time.time()
        for epoch in range(200):
            # print(epoch,strokes.max(),strokes.min(),strokes[0][0])
            output = model.rendering_with_list(new_strokes)
            output = output[ :3, :, :]
            loss = l2_loss(output, ori_img)
            loss.backward()
            # loss_scaler(loss, optimizer, parameters=optimized_strokes[0],
            #             update_grad=True)
            # loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(epoch, loss)
            if epoch % 10 == 0:
                save_image(torch.stack([ori_img.squeeze(), output], dim=0), 'output/%d-%d.jpg' % (idx, epoch), normalize=False)
                print(time.time()-s)

    print(average_loss/idx)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
