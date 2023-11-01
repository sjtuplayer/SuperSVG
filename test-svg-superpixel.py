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
resize_256 = transforms.Resize([256, 256])
resize_224=transforms.Resize([224,224])
@torch.no_grad()
def decode_by_id_map(image,model):
    #slic划分后，按照id_map筛选没在mask中的paths
    width = 512
    paths_per_region =64
    model.width = width
    shift=1
    image = image.resize((width, width))
    image = np.array(image) / 255.0
    segments = slic(image, n_segments=32, sigma=5,compactness=50) # 512*512, 90 segments
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
    dialate_masks=[]
    for i in range(1, segment_num + 1):
        seg_mask = (segments == i).numpy().astype('uint8')
        seg_mask_dialate = cv2.dilate(seg_mask, kernel, iterations=shift)
        seg_mask = torch.from_numpy(seg_mask)
        seg_mask_dialate = torch.from_numpy(seg_mask_dialate > 0.3).int()
        #mask = torch.from_numpy(seg_mask_dialate > 0.3)
        idxs = torch.nonzero(seg_mask_dialate)
        x1, x2, y1, y2 = idxs[:, 0].min(), idxs[:, 0].max(), idxs[:, 1].min(), idxs[:, 1].max()
        #x1, x2, y1, y2 = max(0, x1 - shift), min(width - 1, x2 + shift), max(0, y1 - shift), min(width - 1, y2 + shift)
        coords.append((x1 / h, (x2+1) / h, y1 / w, (y2+1) / w))
        img = image * seg_mask_dialate.unsqueeze(0)
        img = img[:, x1:x2 + 1, y1:y2 + 1]
        mask = seg_mask[x1:x2 + 1, y1:y2 + 1].unsqueeze(0)
        img = resize_224(img)
        dialate_masks.append(resize_224(seg_mask_dialate[x1:x2 + 1, y1:y2 + 1].unsqueeze(0)))
        #print(mask.shape)
        imgs.append(img)
        masks.append(resize_224(mask))


    imgs = torch.stack(imgs, dim=0).cuda().float()
    masks0 = torch.stack(masks, dim=0).cuda().float()
    dialate_masks= torch.stack(dialate_masks, dim=0).cuda().float()
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
    model.width = 224
    output = model.rendering(strokes)[:,:3,:,:]
    imgs = imgs
    output = output
    masks=dialate_masks
    masks=masks0
    print(((output - imgs) ** 2).mean())
    #for i in range(len(output)):
    loss_mse = F.mse_loss(output * masks, imgs * masks) / (masks.sum() / (masks.size(0) * masks.size(-1) * masks.size(-2)))
    # loss_mse2 = F.mse_loss(output * (1-masks), imgs * (1-masks),reduction='sum')
    # print('mask sum',float(masks.sum()),'1-mask sum',float((1-masks).sum()))
    #print(F.mse_loss(output , imgs,reduction='sum'),F.mse_loss(output * masks, imgs * masks,reduction='sum'),loss_mse,loss_mse2)
    loss_mse2 = F.mse_loss(output * (1 - masks), imgs * (1 - masks))/ ((1-masks).sum() / (masks.size(0) * masks.size(-1) * masks.size(-2)))
    print('total mse',float(F.mse_loss(output, imgs)), 'total mse with mask',float(F.mse_loss(output * masks, imgs * masks)),
          'inside mse',float(loss_mse),'outside mse', float(loss_mse2))
    save_image(torch.cat([imgs[:8],output[:8],output[:8]*(1-masks[:8]),masks0[:8].repeat(1,3,1,1),masks[:8].repeat(1,3,1,1)],dim=0),'tmp-0.jpg',nrow=8)
    new_strokes = []
    id_strokes=[]
    cnt=1
    model.width=width
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
        for j in range(len(strokes[i])):
            stroke = strokes[i][j]
            # if id in id_map or id+0.5 in id_map or id-0.5 in id_map or id-0.5 in id_map:
            if id in id_map:
                x1, x2, y1, y2 = coords[i]
                stroke[1:-3:2] = x1 + (x2 - x1) * stroke[1:-3:2]
                stroke[0:-3:2] = y1 + (y2 - y1) * stroke[0:-3:2]
                new_strokes.append(stroke)
            id+=1
    print('num of strokes',len(new_strokes))
    new_strokes = torch.stack(new_strokes, dim=0).unsqueeze(0)
    output = model.rendering(new_strokes)
    output = resize_512(output[:, :3, :, :])
    print(F.mse_loss(output,image.cuda()))
    save_image(output, 'tmp.jpg', normalize=False)
def center_padding(img,mask,x1,x2,y1,y2):
    w=x2-x1+1
    h=y2-y1+1
    if h>w:
        padding1=torch.zeros((3,(h-w)//2,h))
        padding2 = torch.zeros((3,  h-w-(h - w) // 2,h))
        img=torch.cat([padding1,img,padding2],dim=1)
        print(img.shape)
        mask=torch.cat([padding1[0:1,:,:],mask,padding2[0:1,:,:]],dim=1)
        x1=x1-(h-w)//2
        x2=x2+h-w-(h - w) // 2
        print(x2-x1+1,y2-y1+1)
    if h<w:
        padding1=torch.zeros((3,w,(w-h)//2))
        padding2 = torch.zeros((3, w,w-h-(w - h) // 2))
        img=torch.cat([padding1,img,padding2],dim=2)
        print(img.shape)
        mask = torch.cat([padding1[0:1,:,:], mask, padding2[0:1,:,:]], dim=2)
        y1=y1-(w-h)//2
        y2=y2+w-h-(w - h) // 2
        print(x2 - x1 + 1, y2 - y1 + 1)
    return img,mask,x1,x2,y1,y2
@torch.no_grad()
def decode_by_mask(image,model):
    #用每个小块的mask拼接出图片
    width=512
    shift=1
    bs=64
    image=image.resize((width,width))
    image = np.array(image) / 255.0
    segments = slic(image, n_segments=128, sigma=5,compactness=100)
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
    int_coords=[]
    resize_masks=[]
    kernel = np.ones((4, 4), np.uint8)
    for i in range(1, segment_num + 1):
        seg_mask = (segments == i).astype('uint8')
        seg_mask_dialate = cv2.dilate(seg_mask, kernel, iterations=shift)
        seg_mask=torch.from_numpy(seg_mask)
        seg_mask_dialate=torch.from_numpy(seg_mask_dialate>0.3)
        idxs = torch.nonzero(seg_mask_dialate)
        x1, x2, y1, y2 = idxs[:, 0].min(), idxs[:, 0].max(), idxs[:, 1].min(), idxs[:, 1].max()
        #x1,x2,y1,y2=max(0,x1-shift),min(width-1,x2+shift),max(0,y1-shift),min(width-1,y2+shift)
        img = image * seg_mask_dialate.unsqueeze(0)
        img = img[:, x1:x2 + 1, y1:y2 + 1]
        mask = seg_mask
        resize_mask=seg_mask[x1:x2 + 1, y1:y2 + 1].unsqueeze(0)
        img,resize_mask,x1,x2,y1,y2=center_padding(img,resize_mask,x1,x2,y1,y2)
        coords.append((x1 / h, (x2+1) / h, y1 / w, (y2+1) / w))
        int_coords.append((x1, x2, y1, y2))
        #mask=seg_mask_dialate.int()
        img = resize_128(img)
        resize_mask=resize_128(resize_mask)
        imgs.append(img)
        masks.append(mask.cuda())
        resize_masks.append(resize_mask)
    image=image.cuda()
    imgs = torch.stack(imgs, dim=0).cuda().float()
    resize_masks=torch.stack(resize_masks,dim=0).cuda()
    if imgs.size(0)<=bs:
        strokes = model.predict_path(imgs,num=64)  # num*128*27
    else:
        strokes=[]
        for j in range(imgs.size(0)//bs+1):
            strokes.append(model.predict_path(imgs[j*bs:min((j+1)*bs,imgs.size(0))],num=64))
            #print(imgs[j*bs:min((j+1)*bs,img.size(0)-1)].shape,'hh')
        strokes=torch.cat(strokes,dim=0)
    print(strokes.shape,imgs.shape)
    model.width = 128
    output0 = model.rendering(strokes)[:, :3, :, :]
    print('loss in each superpixel',((output0 - imgs) ** 2).mean())
    print(F.mse_loss(output0, imgs), F.mse_loss(output0 * resize_masks, imgs * resize_masks)/(resize_masks.sum()/(resize_masks.size(0)*resize_masks.size(-1)*resize_masks.size(-2))),'hhhh')
    # print(output.min(), output.max(), imgs.min(), imgs.max())
    # save_image(torch.cat([imgs[:8], output[:8]], dim=0), 'tmp-0.jpg', nrow=8)
    model.width=width
    losses=0
    cnt=0
    output_image = torch.zeros(ori_shape).cuda()
    for i in range(len(imgs)):
        x1, x2, y1, y2 = coords[i]
        new_strokes=[]
        ori_strokes=strokes[i].clone()
        model.width=128
        ori_output = model.rendering(ori_strokes.unsqueeze(0))[:,:3,:,:]
        for j in range(len(strokes[i])):
            stroke = strokes[i][j]

            # print(x1,x2,y1,y2)
            stroke[1:-3:2] = x1 + (x2 - x1) * stroke[1:-3:2]
            stroke[0:-3:2] = y1 + (y2 - y1) * stroke[0:-3:2]
            new_strokes.append(stroke)

        new_strokes = torch.stack(new_strokes, dim=0).unsqueeze(0)
        model.width = width
        output = model.rendering(new_strokes)
        output = output[0, :3, :, :]
        mask=masks[i]
        print( F.mse_loss(output * mask, image* mask, reduction='sum') / (mask.sum() * 3),
               F.mse_loss(ori_output[0] * resize_masks[i], imgs[i] * resize_masks[i])/(resize_masks[i].sum()/(resize_masks[i].size(0)*resize_masks[i].size(-1)*resize_masks[i].size(-2))))
        losses += F.mse_loss(output * mask, image* mask, reduction='sum') / (mask.sum() * 3)
        cnt+=1
        output_image = output_image * (1 - mask) + mask * output
        # mask=masks[i].unsqueeze(0)
        # #losses+=F.mse_loss(output*mask,image.cuda()*mask)/(mask.sum()/(mask.size(-1)*mask.size(-2)))
        # # losses += F.mse_loss(output * mask, image* mask, reduction='sum') / (mask.sum() * 3)
        # # cnt+=1
        #
        # #print(losses/cnt,'ggggggggg')
        #
        # idxs = torch.nonzero(mask.squeeze())
        # x1, x2, y1, y2 = int_coords[i]
        # resize = transforms.Resize((x2+1-x1,y2+1-y1))
        # resize_ori_output=resize(ori_output)
        # tmp_canvas=torch.zeros((1,3,512,512)).cuda()
        # tmp_canvas[:,:,x1:x2+1,y1:y2+1]=resize_ori_output
        # #output_image = output_image * (1 - mask) + mask * tmp_canvas
        # output_image=output_image * (1 - mask) + mask * output
        # # losses += F.mse_loss(tmp_canvas * mask, image* mask, reduction='sum') / (mask.sum() * 3)
        # # cnt+=1
        # #tmp_mask=resize_128()
        # tmp_mask=resize_128(mask[:, x1:x2 + 1, y1:y2 + 1])
        # tmp_mask=(tmp_mask>0.3).int()
        # tmp_img = resize_128((image*mask)[:, x1:x2 + 1, y1:y2 + 1])
        # tmp_out=resize_128((output*mask)[:, x1:x2 + 1, y1:y2 + 1])
        # losses += F.mse_loss(tmp_canvas*mask, output*mask,reduction='sum')/(mask.repeat(3,1,1).sum())
        # cnt += 1
        # #print((output*mask).shape,F.mse_loss(output*mask, image*mask,reduction='sum')/(mask.repeat(3,1,1).sum()))
        # #print(F.mse_loss(output0[i], imgs[i]), F.mse_loss(output0[i] * resize_masks[i], imgs[i] * resize_masks[i]),F.mse_loss(imgs[i],tmp_out),F.mse_loss(imgs[i]*resize_masks[i],tmp_out*resize_masks[i]),F.mse_loss(tmp_out,ori_output))
        # # if F.mse_loss(imgs[i],ori_output)<0.005:
        # print('ggg',F.mse_loss(imgs[i],ori_output),resize_ori_output.shape,output[:,x1:x2+1,y1:y2+1].shape)
        # save_image(torch.stack([(image*masks[i])[:,x1:x2+1,y1:y2+1], resize_ori_output[0], output[:,x1:x2+1,y1:y2+1], ((resize_ori_output[0] - output[:,x1:x2+1,y1:y2+1]).abs()>0.2).float()], dim=0),
        #            'tmp1-%d.jpg' % i)
            # save_image(torch.stack([imgs[i],tmp_out,ori_output[0],(tmp_out-ori_output[0])**2],dim=0),'tmp0-%d.jpg'%i)

        #print(F.mse_loss(imgs[i],ori_output))
        # if i==56:
        #     print(F.mse_loss(imgs[i],tmp_img,reduce=False))

        #print(F.mse_loss(imgs[i],ori_output),F.mse_loss(imgs[i]*tmp_mask,ori_output*tmp_mask))
    print('end iter')
    print(losses/cnt)
    print(((output_image - image.cuda()) ** 2).mean())
    save_image(output_image, 'tmp.jpg', normalize=False)
    exit()

def test_resize(image,model):
    resize_128=transforms.Resize([128,128])
    resize_64 = transforms.Resize([64, 64])
    width = 128
    shift = 1
    bs = 64
    image = image.resize((64, 32))
    image = np.array(image) / 255.0
    image=torch.from_numpy(image).cuda().unsqueeze(0).float().permute(0,3,1,2)
    input_image=resize_128(image)
    strokes = model.predict_path(input_image, num=64)
    model.width = 128
    output0 = model.rendering(strokes)[:, :3, :, :]
    print('ori distance',F.mse_loss(input_image, output0))
    strokes[:,:,1:-3:2] = 0.25*strokes[:,:,1:-3:2]
    strokes[:,:,0:-3:2] = 0.5*strokes[:,:,0:-3:2]
    model.width=width
    output=model.rendering(strokes)[:,:3,:32,:64]
    print('final distance',F.mse_loss(image,output))
    tmp_resize=transforms.Resize([32,64])
    output2=tmp_resize(output0)
    print('final distance2', F.mse_loss(image, output2))
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

    # model=AttnPainterSVG(stroke_num=128,path_num=4,width=width,control_num=False)
    # ckpt=torch.load('/home/huteng/SVG/AttnPainter/output-test/checkpoints-best.pt')
    model = AttnPainterSVG(stroke_num=128, path_num=4, width=width, control_num=True)
    ckpt = torch.load('/home/huteng/SVG/AttnPainter/output_superpixel/checkpointsmask_loss7.pt')
    # model = AttnPainterSVG(stroke_num=128, path_num=4, width=width, control_num=False)
    # ckpt=torch.load('/home/huteng/SVG/AttnPainter/output_superpixel/checkpoints128_paths-mask_loss.pt')
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
        if idx==16:
            print(file)
            image = Image.open(file).convert('RGB')
            # test_resize(image,model)
            # exit()
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

