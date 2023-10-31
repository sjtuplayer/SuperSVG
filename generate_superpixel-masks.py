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
data_root='/home/huteng/dataset/imagenet/train'
subdirs=[os.path.join(data_root,i) for i in os.listdir(data_root)]
paths=[]
for subdir in subdirs:
    paths+=[os.path.join(subdir,i) for i in os.listdir(subdir)]
resize=transforms.Resize([width,width])
cnt=0
for i in range(len(paths)):
    if i%100==0:
        print(i,len(paths))
    image = Image.open(paths[i]).convert('RGB')
    image = np.array(image) / 255.0
    segments = torch.from_numpy(slic(image, n_segments=16, sigma=5, compactness=30))
    segment_num = segments.max()
    for mask_idx in range(1,segment_num+1):
        seg_mask = (segments == mask_idx).int()
        idxs = torch.nonzero(seg_mask)
        x1, x2, y1, y2 = idxs[:, 0].min(), idxs[:, 0].max(), idxs[:, 1].min(), idxs[:, 1].max()
        mask = seg_mask[x1:x2 + 1, y1:y2 + 1].unsqueeze(0)
        mask = resize(mask)
        mask = (mask > 0.5).int().float()
        for tmp_mask in mask:
            save_image(tmp_mask,os.path.join('/home/huteng/SVG/AttnPainter/dataset/superpixel-mask','%d.jpg'%cnt),normalize=False)
            cnt+=1
