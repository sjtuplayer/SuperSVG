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
import torch.nn.functional as F
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
def decode_by_id_map(image,model,debug=False):
    #slic划分后，按照id_map筛选没在mask中的paths
    width = 512
    paths_per_region = 64
    model.width = width
    shift=1
    image = image.resize((width, width))
    image = np.array(image) / 255.0
    segments = slic(image, n_segments=32, sigma=5, compactness=30) # 512*512, 90 segments
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
        idxs = torch.nonzero(seg_mask_dialate) # 非零索引
        x1, x2, y1, y2 = idxs[:, 0].min(), idxs[:, 0].max(), idxs[:, 1].min(), idxs[:, 1].max()
        #x1, x2, y1, y2 = max(0, x1 - shift), min(width - 1, x2 + shift), max(0, y1 - shift), min(width - 1, y2 + shift)
        coords.append((x1 / h, x2 / h, y1 / w, y2 / w))
        img = image * seg_mask_dialate.unsqueeze(0)
        img = img[:, x1:x2 + 1, y1:y2 + 1]
        mask=seg_mask_dialate[x1:x2 + 1, y1:y2 + 1].unsqueeze(0)
        #mask = seg_mask[x1:x2 + 1, y1:y2 + 1].unsqueeze(0)
        img = resize_128(img)
        #print(mask.shape)
        imgs.append(img)
        masks.append(resize_512(mask))
        # masks.append(mask)
    imgs = torch.stack(imgs, dim=0).cuda().float() # [90, 3, 128, 128] 90张图
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

    output = model.rendering(strokes[7:8])
    output = output[:, :3, :, :]
    save_image(output, 'tmp-ori.jpg', normalize=False)
    id_stroke=strokes[0:1].clone()
    for j in range(id_stroke.size(1)):
        id_stroke[:,j,-3:]=j+1
    ori_id_map = model.rendering(id_stroke)
    ori_id_map = ori_id_map[0, 0, :, :]
    new_strokes0 = []

    for i in range(len(imgs)):
        print(i)
        if i!=7:
            continue
            break
        # id_map = model.rendering(id_strokes)
        # id_map = id_map[0, 0, :, :]
        mask = (masks[i] == 1).int()
        mask_array = mask.cpu().numpy()
        mask_array = (mask_array * 255).astype(np.uint8)
        mask_image = mask_array.squeeze(0)
        # print(i,j,id_strokes.size(1))
        cv2.imwrite('tmp_mask.jpg', mask_image)
        # 要把外面的点均匀放在边界上
        edge, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        edge = edge[0]
        filtered_edge = [point[0] for index,point in enumerate(edge) if point[0][0] != 0 and point[0][1] != 0 or (edge[(index+1)%len(edge)][0][0] != 0 and edge[(index+1)%len(edge)][0][1] != 0) or (edge[(index-1)%len(edge)][0][0] != 0 and edge[(index-1)%len(edge)][0][1] != 0)] # array
        filtered_edge=torch.tensor(filtered_edge)
        mask_boundary=torch.from_numpy(mask_array).repeat(3,1,1).float()
        mask_boundary=torch.zeros_like(mask_boundary)
        for point in filtered_edge:
            mask_boundary[:,point[1],point[0]]=1
        # save_image(mask_image,'tmp_mask-boundary.jpg')
        new_strokes=[]
        debug=True
        for j in range(0, len(strokes[i])): # 128
            save_images=[]
            # if j==70:
            #     debug=True
            # else:
            #     debug=False
            if j+1 not in ori_id_map:
                continue
            stroke = strokes[i][j] # [y1, x1, y2, x2, ..., y12, x12, r, g, b]
            # if j not in id_map:
            #     continue
            id_stroke = stroke.clone().unsqueeze(0).unsqueeze(0) # [1, 1, 27]
            id_stroke[0, 0,-3:] = j
            id_map = model.rendering(id_stroke) # 只渲染一条stroke
            output_tmp = id_map[:, :3, :, :]

            id_map = id_map[0, 0, :, :]
            id_map1 = mask[0] * id_map # 有用的
            id_map0 = (1 - mask[0]) * id_map # 去掉的
            ori_id_map1 = mask[0] * (ori_id_map==j+1).int()  # 有用的
            ori_id_map0 = (1 - mask[0]) * (ori_id_map==j+1).int()  # 去掉的
            if (ori_id_map1!=0).int().sum()<100:
                 continue
            if (id_map1 == j).sum()<200: continue
            new_stroke = stroke.clone()
            if (id_map0 == j).sum() != 0 and (id_map1 == j).sum() != 0: # stroke经过边界,但不一定有控制点在边界外
                p_cnt = 0 # 黑色区域点数
                curr_stroke = []
                flag=False
                inside_flag=[]
                for k in range(int((len(stroke)-3)/2)): # 12
                    y = int((stroke[2*k] * w).item())
                    x = int((stroke[2*k+1] * h).item())
                    if mask[0][x][y]==1: 
                        # curr_stroke.append(y/w)
                        # curr_stroke.append(x/h)
                        inside_flag.append(True)
                        continue
                    inside_flag.append(False)
                    p_cnt += 1
                    if k%3==0:
                        flag=True
                if p_cnt != 0 and flag and p_cnt!=4:
                    mask_array = (id_map!=0).cpu().numpy()
                    mask_array = (mask_array * 255).astype(np.uint8)
                    mask_image = mask_array
                    if debug:
                        #cv2.imwrite('tmp_mask.jpg', mask_image)
                        print(mask_image.max(),mask_image.min())
                    edge, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    edge = edge[0]
                    if len(edge)<10:
                        continue
                    path_edge = [point[0] for point in edge]  # array
                    path_edge=torch.tensor(path_edge)
                    #stroke boundary
                    mask_image = torch.from_numpy(mask_array).repeat(3, 1, 1).float()
                    mask_image = torch.ones_like(mask_image)
                    if debug:
                        print(path_edge,len(path_edge),'hhhhhhhhhhhh',edge)
                    for point in path_edge:
                        mask_image[:,point[1],point[0]]=0
                    tmp_l = 5
                    if debug:  #画边界
                        for k in range(int((len(stroke) - 3) / 2)):  # 12
                            y = int((stroke[2 * k] * w).item())
                            x = int((stroke[2 * k + 1] * h).item())
                            print(x,y)
                            if k % 3 != 0:
                                mask_image[:, max(x - tmp_l, 0):min(512, x + tmp_l),
                                max(y - tmp_l, 0):min(512, y + tmp_l)] = 0
                                mask_image[1, max(x - tmp_l, 0):min(512, x + tmp_l),
                                max(y - tmp_l, 0):min(512, y + tmp_l)] = 1
                                continue
                            mask_image[:, max(x - tmp_l, 0):min(512, x + tmp_l), max(y - tmp_l, 0):min(512, y + tmp_l)] = 0
                            mask_image[0, max(x - tmp_l, 0):min(512, x + tmp_l), max(y - tmp_l, 0):min(512, y + tmp_l)] = 1
                        mask_image=mask_image-mask_boundary
                        save_images.append(mask_image.cpu().clone())
                        #save_image(mask_image,'tmp-stroke_boundary.jpg')

                    end_point_index=[]
                    for k in range(4):
                        kk=k*3
                        #if inside_flag[k]:
                        y = int((stroke[2 * kk] * w).item())
                        x = int((stroke[2 * kk + 1] * h).item())
                        coord=torch.tensor([[y,x]])
                        index=((coord-path_edge)**2).sum(dim=1).argmin()
                        end_point_index.append(index)
                    need_reverse_find = False
                    if debug: print(end_point_index)
                    tmp_point_index=[]
                    for k in range(4):
                        index=(end_point_index[k]-end_point_index[0]+len(path_edge))%len(path_edge)
                        tmp_point_index.append(index)
                    if tmp_point_index[2]<tmp_point_index[1]:
                        need_reverse_find = True
                    for k in range(4):
                        y,x=path_edge[end_point_index[k]]
                        y2,x2=path_edge[end_point_index[(k+1)%4]]
                        if debug:print('k',mask[0][x][y]==1,mask[0][x2][y2]==0)
                        if mask[0][x][y]==1 and mask[0][x2][y2]==0:
                            if debug:
                                print('k',k,(k + 1) % 4,(k + 2) % 4,(k + 3) % 4,need_reverse_find)
                            l=1
                            y2, x2 = path_edge[end_point_index[(k + 2) % 4]]
                            if mask[0][x2][y2] == 0:
                                l=2
                                y2, x2 = path_edge[end_point_index[(k + 3) % 4]]
                                if mask[0][x2][y2] == 0:
                                    l=3
                            if debug: print('k=',k,'l=',l)
                            start_end_point_idx=end_point_index[k]   #端点在贝塞尔曲线中的id
                            end_endpoint_idx=end_point_index[(k+l)%4]  #端点在贝塞尔曲线中的id
                            if debug: print('start id in besaier',start_end_point_idx,'end id in besaier',end_endpoint_idx)
                            for kk in range(len(path_edge)):
                                if not need_reverse_find:
                                    tmp_idx=(start_end_point_idx+kk)%len(path_edge)
                                    y2, x2 = path_edge[(tmp_idx + 1) % len(path_edge)]
                                else:
                                    tmp_idx = (start_end_point_idx - kk+len(path_edge)) % len(path_edge)
                                    y2, x2 = path_edge[(tmp_idx + -1+len(path_edge)) % len(path_edge)]

                                if mask[0,x,y]==1 and mask[0,x2,y2]==0:
                                    start_idx=tmp_idx
                                    break
                            for kk in range(len(path_edge)):
                                if not need_reverse_find:
                                    tmp_idx=(end_endpoint_idx+kk)%len(path_edge)
                                    y2, x2 = path_edge[(tmp_idx + 1) % len(path_edge)]
                                else:
                                    tmp_idx = (end_endpoint_idx - kk+len(path_edge)) % len(path_edge)
                                    y2, x2 = path_edge[(tmp_idx - 1+len(path_edge)) % len(path_edge)]
                                y,x=path_edge[tmp_idx]

                                #print(tmp_idx,int(y), int(x), int(mask[0, x, y]),int(mask[0,x2,y2]))
                                if mask[0,x,y]==0 and mask[0,x2,y2]==1:
                                    end_idx=tmp_idx
                                    break
                            y,x=path_edge[start_idx]
                            if debug:
                                mask_image[:, max(x - tmp_l, 0):min(512, x + tmp_l),
                                max(y - tmp_l, 0):min(512, y + tmp_l)] = 0
                                mask_image[2, max(x - tmp_l, 0):min(512, x + tmp_l),
                                max(y - tmp_l, 0):min(512, y + tmp_l)] = 1
                                y, x = path_edge[end_idx]
                                mask_image[:, max(x-tmp_l,0) :min(512,x + tmp_l), max(y-tmp_l,0) :min(512,y + tmp_l)] = 0
                                mask_image[2, max(x-tmp_l,0) :min(512,x + tmp_l), max(y-tmp_l,0) :min(512,y + tmp_l)] = 1
                                save_images.append(mask_image.cpu().clone())
                                #save_image(mask_image, 'tmp-stroke_boundary.jpg')
                            start_coord=path_edge[start_idx].unsqueeze(0)
                            end_coord = path_edge[end_idx].unsqueeze(0)
                            if debug: print(start_coord,end_coord)
                            start_idx = ((start_coord - filtered_edge) ** 2).sum(dim=1).argmin()
                            end_idx = ((end_coord - filtered_edge) ** 2).sum(dim=1).argmin()
                            if debug: print(start_idx, end_idx,l,k)
                            if l>1:
                                for kk in range(2):
                                    kk_idx = start_idx
                                    y, x = filtered_edge[kk_idx]
                                    kk_idx = (k*3 + kk + 1) % 12
                                    new_stroke[kk_idx * 2] = y / 512
                                    new_stroke[kk_idx * 2 + 1] = x / 512
                                for kk in range(2):
                                    kk_idx = end_idx
                                    y, x = filtered_edge[kk_idx]
                                    kk_idx = (k*3 + l*3+kk+1) % 12
                                    new_stroke[kk_idx * 2] = y / 512
                                    new_stroke[kk_idx * 2 + 1] = x / 512
                                for kk in range(2,l*3):
                                    kk_idx=start_idx+int((end_idx-start_idx)/(l*3-3)*(kk-2))
                                    y, x = filtered_edge[kk_idx]
                                    kk_idx=(k*3+kk+1)%12
                                    new_stroke[kk_idx*2]=y/512
                                    new_stroke[kk_idx * 2+1] = x/512
                            else:
                                for kk in range(l*3+2):
                                    kk_idx=start_idx+int((end_idx-start_idx)/(l*3+1)*kk)
                                    y, x = filtered_edge[kk_idx]
                                    kk_idx=(k+kk+1)%12
                                    new_stroke[kk_idx*2]=y/512
                                    new_stroke[kk_idx * 2+1] = x/512
                    if debug:
                        output_tmp = model.rendering(new_stroke.unsqueeze(0).unsqueeze(0))  # 只渲染一条stroke
                        output_tmp = output_tmp[0, :3, :, :]
                        stroke = new_stroke.clone()
                        for k in range(int((len(stroke) - 3) / 2)):  # 12
                            y = int((stroke[2 * k] * w).item())
                            x = int((stroke[2 * k + 1] * h).item())
                            if k % 3 != 0:
                                tmp_l = 5
                                output_tmp[:, max(x - tmp_l, 0):min(512, x + tmp_l),
                                max(y - tmp_l, 0):min(512, y + tmp_l)] = 0
                                output_tmp[1, max(x - tmp_l, 0):min(512, x + tmp_l),
                                max(y - tmp_l, 0):min(512, y + tmp_l)] = 1
                                continue
                            tmp_l = 7
                            output_tmp[:, max(x - tmp_l, 0):min(512, x + tmp_l),
                            max(y - tmp_l, 0):min(512, y + tmp_l)] = 0
                            output_tmp[0, max(x - tmp_l, 0):min(512, x + tmp_l),
                            max(y - tmp_l, 0):min(512, y + tmp_l)] = 1
                        #save_image(mask_image, 'tmp-stroke_boundary-after.jpg')
                        #save_image(output_tmp, 'tmp-new0-after.jpg', normalize=False)
                        save_images.append(output_tmp.cpu())
                        save_images=torch.stack(save_images,dim=0)
                        save_image(save_images,'output/tmp-all-%d.jpg'%j)
            new_strokes.append(new_stroke.clone())
            if debug:
                tmp_new_strokes=torch.stack(new_strokes,dim=0)
                output_tmp = model.rendering(tmp_new_strokes.unsqueeze(0).cuda())  # 只渲染一条stroke
                output_tmp = output_tmp[0, :3, :, :]
                save_image(output_tmp, 'output/tmp-stroke-%d.jpg' % j)
        new_strokes=torch.stack(new_strokes,dim=0)
        #if debug:
        output_tmp = model.rendering(new_strokes.unsqueeze(0).cuda()) # 只渲染一条stroke
        output_tmp = output_tmp[:, :3, :, :]
        save_image(output_tmp, 'output/tmp-img-%d.jpg'%i, normalize=False)
            #new_strokes.append(stroke.cuda())
        new_strokes0.append(new_strokes)
###
    print(len(new_strokes0))
    strokes = new_strokes0
    new_strokes=[]
    for i in range(len(strokes)):
        for j in range(len(strokes[i])):
            stroke = strokes[i][j]
            # if id in id_map or id+0.5 in id_map or id-0.5 in id_map or id-0.5 in id_map:
            #if id in id_map:
            if True:
            #if id in ids:
                x1, x2, y1, y2 = coords[i]
                stroke[1:-3:2] = x1 + (x2 - x1) * stroke[1:-3:2]
                stroke[0:-3:2] = y1 + (y2 - y1) * stroke[0:-3:2]
                new_strokes.append(stroke)
    print('num of strokes',len(new_strokes))
    new_strokes = torch.stack(new_strokes, dim=0).unsqueeze(0)
    output = model.rendering(new_strokes)
    output = resize_512(output[:, :3, :, :])
    print(F.mse_loss(output,image.cuda()))
    save_image(output, 'tmp.jpg', normalize=False)
@torch.no_grad()
def decode_by_mask(image,model):
    #用每个小块的mask拼接出图片
    width=512
    shift=1
    bs=64
    image=image.resize((width,width))
    image = np.array(image) / 255.0
    segments = slic(image, n_segments=128, sigma=5)
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
        # coords.append((x1, x2, y1, y2))
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
            #decode_by_mask(image, model)
            decode_by_id_map(image,model)
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

