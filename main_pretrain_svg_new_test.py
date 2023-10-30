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
from models.encoder import StrokeAttentionPredictor_autoregression

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from models.attn_painter import AttnPainterSVG as AttnPainterSVG_normal
from models.attn_painter_autoregression import AttnPainterSVG as AttnPainterSVG_autoregression
from PIL import Image
from torchvision.utils import save_image

width=256
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
    # Model parameters
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=0.002, metavar='LR',
                        help='learning rate (absolute lr)')
    # Dataset parameters
    parser.add_argument('--data_path', default='/home/huteng/dataset/imagenet/val', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir0',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=1, type=int)

    return parser


def main(args):
    device = torch.device(args.device)
    dataset_train = AttrImgDataset(args.data_path)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True
    )
    
    # define the model
    model1=AttnPainterSVG_normal(stroke_num=128,path_num=4,width=width)
    ckpt=torch.load('/home/huteng/SVG/AttnPainter/output-test/checkpoints.pt')
    model1.load_state_dict(ckpt,strict=False)
    model2=AttnPainterSVG_autoregression(stroke_num=128,path_num=4,width=width)
    model2.encoder = StrokeAttentionPredictor_autoregression(stroke_num=128, stroke_dim=4 * 6 + 3,control_num=False)
    model2.load_state_dict(torch.load('/home/huteng/SVG/AttnPainter/output_dir0/checkpoints.pt'),strict=False)


    for i, input1 in enumerate(data_loader_train):
        with torch.no_grad():

            output1, _ = model1(input1)
            output1 = output1[:,:3,:,:].detach()
            output1 = output1.cpu()
            input2 = torch.cat([input1, output1], dim=1)
            output2, _ = model2(input2)
            output2 = output2[:,:3,:,:].cpu()
            save_image(input1, 'tmp-in1.jpg', normalize=False)
            save_image(output1, 'tmp-out1.jpg', normalize=False)
            save_image(output2, 'tmp-out2.jpg', normalize=False)
        break


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
