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
import skimage
import skimage.io
import pydiffvg

from util.misc import NativeScalerWithGradNormCount as NativeScaler

from models.attn_painter import AttnPainterSVG
from PIL import Image

width=128

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
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=0.02, metavar='LR',
                        help='learning rate (absolute lr)')

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

    parser.set_defaults(pin_mem=True)

    return parser


def main(args):
    device = torch.device(args.device)
    model=AttnPainterSVG(stroke_num=128,path_num=4,width=width)
    ckpt=torch.load('/home/huteng/SVG/AttnPainter/output-test/checkpoints.pt')
    model.load_state_dict(ckpt)
    model.to(device)

    loss_scaler = NativeScaler()

    files=os.listdir(r"/home/huteng/dataset/imagenet/val/")
    model.eval()
    l2_loss=torch.nn.MSELoss()
    from torchvision.utils import save_image
    average_loss=0
    for idx, img in enumerate(files):
        img = "/home/huteng/dataset/imagenet/val/"+img
        target = torch.from_numpy(skimage.io.imread(img)).to(torch.float32) / 255.0
        target = target.unsqueeze(0)
        target = target.permute(0, 3, 1, 2)

        strokes = torch.rand(target.size()[2]*target.size()[3], 27).detach().cuda()
        strokes = strokes.unsqueeze(0)
        strokes.requires_grad=True
        optimizer=torch.optim.Adam([strokes], lr=0.001, betas=(0.9, 0.95))

        for epoch in range(250):
            #print(epoch,strokes.max(),strokes.min(),strokes[0][0])
            output=model.rendering(strokes, save_svg_path='output/%d.svg' % idx)
            output = output[:, :3, :, :]
            loss=l2_loss(output, target)
            loss_scaler(loss, optimizer, parameters=strokes,
                        update_grad=True)
            # loss.backward()
            # optimizer.step()
            optimizer.zero_grad()
            print(epoch, loss)
            # if epoch%10==0:
            #     save_image(output, 'output/%d-%d.jpg' % (idx,epoch), normalize=False)
        average_loss += float(l2_loss(output, target))
        # output = output.resize(1, block, block, 3, width, width)
        # output = output.permute(0, 3, 1, 4, 2, 5)
        # output = output.resize(1, 3, width * block, width * block)
        save_image(output, 'output/%d.jpg' % idx, normalize=False)
    print(average_loss/idx)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
