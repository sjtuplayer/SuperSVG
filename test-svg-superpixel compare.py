import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from torchvision.utils import save_image
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import timm
from skimage.segmentation import mark_boundaries
# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from models.attn_painter_superpixel import AttnPainterSVG
from PIL import Image
from engine_pretrain import train_one_epoch
from skimage.segmentation import slic
import cv2
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

resize_128 = transforms.Resize([128, 128])
@torch.no_grad()
def decode_by_id_map(image,model):
    #slic划分后，按照id_map筛选没在mask中的paths
    width = 512
    paths_per_region = 64
    model.width = width
    shift=1
    image = image.resize((width, width))
    image = np.array(image) / 255.0
    segments = slic(image, n_segments=128, sigma=5,compactness=5) # 512*512, 90 segments
    print('num_of_segments',segments.max())
    new_img = mark_boundaries(image, segments)
    resize_512 = transforms.Resize([512, 512])
    new_img = torch.from_numpy(new_img).permute(2, 0, 1)
    save_image(resize_512(new_img), 'tmp-slic.jpg')
    segments = torch.from_numpy(segments)
    image = torch.from_numpy(image).permute(2, 0, 1)
    _, h, w = image.size()
    segment_num = segments.max()
    imgs = []
    masks = []
    coords = []
    kernel = np.ones((4, 4), np.uint8)
    for i in range(1, segment_num + 1):
        seg_mask = (segments == i).numpy().astype('uint8')
        seg_mask_dialate = cv2.dilate(seg_mask, kernel, iterations=shift)
        seg_mask = torch.from_numpy(seg_mask)
        seg_mask_dialate = torch.from_numpy(seg_mask_dialate > 0.3).int()
        #mask = torch.from_numpy(seg_mask_dialate > 0.3)
        idxs = torch.nonzero(seg_mask_dialate)
        x1, x2, y1, y2 = idxs[:, 0].min(), idxs[:, 0].max(), idxs[:, 1].min(), idxs[:, 1].max()
        #x1, x2, y1, y2 = max(0, x1 - shift), min(width - 1, x2 + shift), max(0, y1 - shift), min(width - 1, y2 + shift)
        coords.append((x1 / h, x2 / h, y1 / w, y2 / w))
        img = image * seg_mask_dialate.unsqueeze(0)
        img = img[:, x1:x2 + 1, y1:y2 + 1]
        mask = seg_mask[x1:x2 + 1, y1:y2 + 1].unsqueeze(0)
        img = resize_128(img)
        #print(mask.shape)
        imgs.append(img)
        masks.append(resize_128(mask))
    imgs = torch.stack(imgs, dim=0).cuda().float()
    masks = torch.stack(masks, dim=0).cuda().float()
    bs=64

    if imgs.size(0) <= bs:
        strokes = model.predict_path(imgs, num=paths_per_region)  #bs*128*27
    else:
        strokes = []
        for j in range(imgs.size(0) // bs + 1):
            strokes.append(model.predict_path(imgs[j * bs:min((j + 1) * bs, imgs.size(0))], num=paths_per_region))
            # print(imgs[j*bs:min((j+1)*bs,img.size(0)-1)].shape,'hh')
        strokes = torch.cat(strokes, dim=0)
        print(strokes.shape, imgs.shape)
    print(strokes.size(0) * strokes.size(1))
    new_strokes = []
    id_strokes=[]
    cnt=1
    for i in range(len(imgs)):
        tmp_strokes = strokes[i].clone()
        for j in range(tmp_strokes.size(0)):
            stroke = tmp_strokes[j]
            x1, x2, y1, y2 = coords[i]
            stroke[1:-3:2] = x1 + (x2 - x1) * stroke[1:-3:2]
            stroke[0:-3:2] = y1 + (y2 - y1) * stroke[0:-3:2]
            stroke[-3:] = cnt
            id_strokes.append(stroke)
            cnt+=1
    id_strokes=torch.stack(id_strokes,dim=0).unsqueeze(0)
    id_map = model.rendering(id_strokes)
    id_map = id_map[0, 0, :, :]
    id=1
    for i in range(len(imgs)):
        print(i)
        for j in range(len(strokes[i])):
            stroke = strokes[i][j]
            # if id in id_map or id+0.5 in id_map or id-0.5 in id_map or id-0.5 in id_map:
            if id in id_map:
            #if True:
            #if id in ids:
                x1, x2, y1, y2 = coords[i]
                stroke[1:-3:2] = x1 + (x2 - x1) * stroke[1:-3:2]
                stroke[0:-3:2] = y1 + (y2 - y1) * stroke[0:-3:2]
                new_strokes.append(stroke)
            id+=1
    # for i in range(len(imgs)):
    #     id_strokes = strokes[i].clone().unsqueeze(0)
    #     for j in range(id_strokes.size(1)):
    #         id_strokes[0, j, -3:] = j
    #     id_map = model.rendering(id_strokes)
    #     id_map = id_map[0, 0, :, :]
    #     mask = (masks[i] == 1).int()
    #     id_map1 = mask * id_map
    #     id_map0 = (1 - mask) * id_map
    #     # print(i,j,id_strokes.size(1))
    #     for j in range(len(strokes[i])):
    #         stroke = strokes[i][j]
    #         # if j not in id_map:
    #         #     continue
    #         if (id_map0 == j).sum() > (id_map1 == j).sum():
    #             continue
    #         x1, x2, y1, y2 = coords[i]
    #         # print(x1,x2,y1,y2)
    #         stroke[1:-3:2] = x1 + (x2 - x1) * stroke[1:-3:2]
    #         stroke[0:-3:2] = y1 + (y2 - y1) * stroke[0:-3:2]
    #         new_strokes.append(stroke)
    print(len(new_strokes))
    new_strokes = torch.stack(new_strokes, dim=0).unsqueeze(0)

    output = model.rendering(new_strokes)
    print(output.shape, output.min(), output.max())
    output = output[:, :3, :, :]
    print(((output-image.cuda())**2).mean())
    print(output.max(),output.min())
    save_image(output, 'tmp.jpg', normalize=False)
@torch.no_grad()
def decode_by_mask(image,model):
    #用每个小块的mask拼接出图片
    width=512
    shift=1
    bs=64
    image=image.resize((width,width))
    image = np.array(image) / 255.0
    segments = slic(image, n_segments=64, sigma=5)
    print('num_of_segments', segments.max())
    new_img = mark_boundaries(image, segments)
    resize_512 = transforms.Resize([512, 512])
    new_img = torch.from_numpy(new_img).permute(2, 0, 1)
    save_image(resize_512(new_img), 'tmp-slic.jpg')
    #segments = torch.from_numpy(segments)
    image = torch.from_numpy(image).permute(2, 0, 1)
    ori_shape=image.size()
    _, h, w = image.size()
    segment_num = segments.max()
    imgs = []
    masks = []
    coords = []
    kernel = np.ones((4, 4), np.uint8)
    for i in range(1, segment_num + 1):
        seg_mask = (segments == i).astype('uint8')
        seg_mask_dialate = cv2.dilate(seg_mask, kernel, iterations=shift)
        seg_mask=torch.from_numpy(seg_mask)
        seg_mask_dialate=torch.from_numpy(seg_mask_dialate>0.3)
        idxs = torch.nonzero(seg_mask)
        x1, x2, y1, y2 = idxs[:, 0].min(), idxs[:, 0].max(), idxs[:, 1].min(), idxs[:, 1].max()
        x1,x2,y1,y2=max(0,x1-shift),min(width-1,x2+shift),max(0,y1-shift),min(width-1,y2+shift)
        coords.append((x1 / h, x2 / h, y1 / w, y2 / w))
        img = image * seg_mask_dialate.unsqueeze(0)
        img = img[:, x1:x2 + 1, y1:y2 + 1]
        mask = seg_mask
        img = resize_128(img)
        imgs.append(img)
        masks.append(mask.cuda())
    imgs = torch.stack(imgs, dim=0).cuda().float()
    if imgs.size(0)<=bs:
        strokes = model.predict_path(imgs,num=64)  # num*128*27
    else:
        strokes=[]
        for j in range(imgs.size(0)//bs+1):
            strokes.append(model.predict_path(imgs[j*bs:min((j+1)*bs,imgs.size(0))],num=64))
            #print(imgs[j*bs:min((j+1)*bs,img.size(0)-1)].shape,'hh')
        strokes=torch.cat(strokes,dim=0)
        print(strokes.shape,imgs.shape)
    output_image = torch.zeros(ori_shape).cuda()
    for i in range(len(imgs)):
        new_strokes=[]
        for j in range(len(strokes[i])):
            stroke = strokes[i][j]
            x1, x2, y1, y2 = coords[i]
            # print(x1,x2,y1,y2)
            stroke[1:-3:2] = x1 + (x2 - x1) * stroke[1:-3:2]
            stroke[0:-3:2] = y1 + (y2 - y1) * stroke[0:-3:2]
            new_strokes.append(stroke)
        new_strokes = torch.stack(new_strokes, dim=0).unsqueeze(0)
        model.width = width
        output = model.rendering(new_strokes)
        output = output[0, :3, :, :]
        mask=masks[i].unsqueeze(0)
        output_image=output_image*(1-mask)+mask*output
    print(((output_image - image.cuda()) ** 2).mean())
    save_image(output_image, 'tmp.jpg', normalize=False)
    exit()

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

    model=AttnPainterSVG(stroke_num=128,path_num=4,width=width,control_num=False)
    ckpt=torch.load('/home/huteng/SVG/AttnPainter/output-test/checkpoints.pt')
    #ckpt = torch.load('/home/huteng/SVG/AttnPainter/output_superpixel/checkpointsmask_loss2.pt')
    model.load_state_dict(ckpt)

    # checkpoint_model = torch.load('renderer-oil-FCN.pkl', map_location='cpu')
    #
    # msg = model.render.load_state_dict(checkpoint_model, strict=False)
    model.to(device)

    model_without_ddp = model
    #print("Model = %s" % str(model_without_ddp))

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
    name='art'
    dir='/home/huteng/SVG/AttnPainter/test-data/%s'%name
    files=[os.path.join(dir,i) for i in os.listdir(dir)]
    model.eval()
    l2_loss=torch.nn.MSELoss()

    average_loss=0

    for idx,file in enumerate(files):
        if idx==15:
            image = Image.open(file).convert('RGB')
            decode_by_mask(image, model)
            # decode_by_id_map(image,model)
            exit()
        #decode_by_mask(image,model)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

