import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from .encoder import StrokeAttentionPredictor, StrokeAttentionPredictorDen
from .render import FCN
# from .render_oil import FCNOil
import pydiffvg
import lpips
from torch.multiprocessing import Process, Queue
from util import SVR_render
from torchvision import transforms
import random
from util import mophology
channel_mean = torch.tensor([0.485, 0.456, 0.406])
channel_std = torch.tensor([0.229, 0.224, 0.225])
pydiffvg.set_print_timing(False)
pydiffvg.set_use_gpu(True)
MEAN = [-mean / std for mean, std in zip(channel_mean, channel_std)]
STD = [1 / std for std in channel_std]
torch.multiprocessing.set_start_method('spawn', force=True)


class AttnPainterSVG(nn.Module):
    def __init__(self, stroke_num=128, path_num=4, width=128,control_num=False):
        super(AttnPainterSVG, self).__init__()
        self.control_num=control_num
        self.path_num = path_num
        self.encoder = StrokeAttentionPredictor(stroke_num=stroke_num, stroke_dim=path_num * 6 + 3,control_num=control_num)
        self.stroke_num = stroke_num
        self.device = 'cuda'
        self.render = SVR_render.SVGObject(size=(width, width))
        self.width = width
        self.loss_fn_vgg = None

    def forward(self, x,mask=None, num=None,**kwargs):
        if mask is not None:
            x=x*mask-(1-mask)
        if self.control_num:
            if num is None:
                num=random.randint(1,64)
            strokes = self.encoder(x,num)
        else:
            strokes = self.encoder(x)
        # strokes=strokes[:,64:,:]

        pred = self.rendering(strokes, **kwargs)

        return pred, x
    def predict_path(self, x, num=None,**kwargs):
        if self.control_num:
            if num is None:
                num=random.randint(1,64)
            strokes = self.encoder(x,num)
        else:
            strokes = self.encoder(x)

        return strokes
    # def predict_path(self, x):
    #     strokes = self.encoder(x)
    #
    #     return strokes
    def rendering_with_differnent_num(self, strokes, save_svg_path=None, result_queue=None, idx=None):
        shapes = []
        groups = []
        for num in range(len(strokes)):
            num_control_points = [2] * (len(strokes[num][:-3])//6)
            shapes.append(
                pydiffvg.Path(
                    num_control_points=torch.LongTensor(num_control_points),
                    points=strokes[num][:-3].reshape(-1, 2) * self.width,
                    stroke_width=torch.tensor(0.0),
                    is_closed=True))
            groups.append(
                pydiffvg.ShapeGroup(
                    shape_ids=torch.LongTensor([num]),
                    fill_color=torch.cat([strokes[num][-3:], torch.ones(1).cuda()], dim=0)))
        scene_args = pydiffvg.RenderFunction.serialize_scene(self.width, self.width, shapes, groups)
        _render = pydiffvg.RenderFunction.apply
        img = _render(self.width, self.width, 2, 2, 0, None, *scene_args)
        if save_svg_path is not None:
            pydiffvg.save_svg(save_svg_path, self.width, self.width, shapes, groups)
        if result_queue is not None:
            result_queue.put((idx, img))
        return img.permute(2, 0, 1)

    def rendering(self, strokes, save_svg_path=None, result_queue=None, idx=None):
        imgs = []
        strokes = torch.cat([strokes, torch.ones(strokes.size(0), strokes.size(1), 1).to(strokes.device)], dim=2)
        num_control_points = [2] * self.path_num
        for b in range(strokes.size(0)):
            shapes = []
            groups = []
            for num in range(strokes.size(1)):
                shapes.append(
                    pydiffvg.Path(
                        num_control_points=torch.LongTensor(num_control_points),
                        points=strokes[b][num][:-4].reshape(-1, 2) * self.width,
                        stroke_width=torch.tensor(0.0),
                        is_closed=True))
                groups.append(
                    pydiffvg.ShapeGroup(
                        shape_ids=torch.LongTensor([num]),
                        fill_color=strokes[b][num][-4:]))
            scene_args = pydiffvg.RenderFunction.serialize_scene(self.width, self.width, shapes, groups)
            _render = pydiffvg.RenderFunction.apply
            img = _render(self.width, self.width, 2, 2, 0, None, *scene_args)
            imgs.append(img.permute(2, 0, 1))
        imgs = torch.stack(imgs, dim=0)
        if save_svg_path is not None:
            pydiffvg.save_svg(save_svg_path, self.width, self.width, shapes, groups)
        if result_queue is not None:
            result_queue.put((idx, imgs))
        return imgs
    def rendering_batch_one_time(self, strokes,save_svg_path=None): #把batch的paths全部渲染在一张图上，然后再拆开
        block_width=6
        imgs=[]
        strokes=torch.cat([strokes,torch.ones(strokes.size(0),strokes.size(1),1).to(strokes.device)],dim=2)
        num_control_points = [2]*self.path_num
        shapes = []
        groups = []
        for i in range(strokes.size(0)//block_width):
            for j in range(block_width):
                idx=i*block_width+j
                for num in range(strokes.size(1)):
                    points=strokes[idx][num][:-4].reshape(-1, 2)
                    points[:,0]=points[:,0]*self.width+self.width*j
                    points[:,1]=points[:,1]*self.width+self.width*i
                    #print(i,j,points.max(),points.min())
                    shapes.append(
                        pydiffvg.Path(
                            num_control_points=torch.LongTensor(num_control_points),
                            points=points,
                            stroke_width=torch.tensor(0.0),
                            is_closed=True))
                    groups.append(
                        pydiffvg.ShapeGroup(
                            shape_ids=torch.LongTensor([idx*strokes.size(1)+num]),
                            fill_color=strokes[idx][num][-4:]))
        scene_args = pydiffvg.RenderFunction.serialize_scene(self.width*block_width, self.width*block_width, shapes, groups)
        _render = pydiffvg.RenderFunction.apply
        img = _render(self.width*block_width, self.width*block_width, 2, 2, 0, None, *scene_args)
        img=img.permute((2,0,1))[:3,:,:]
        img=img.resize(3,block_width,self.width,block_width,self.width)
        img=img.permute(1,3,0,2,4)
        img=img.resize(strokes.size(0),3,self.width,self.width)
        return img
    def forward_parallel(self, x, **kwargs):
        strokes = self.encoder(x)
        process_list = []
        thread_num = 8
        data_per_thread = x.size(0) // thread_num
        for i in range(thread_num):
            p = Process(target=self.rendering, args=(strokes[data_per_thread * i:process_list * (i + 1)]))
            results = p.start()
            print(results)
            process_list.append(p)

        for p in process_list:
            p.join()
        pred = None
        return pred, x

    def fast_rendering(self, strokes, save_svg_path=None): #TopK渲染方法
        imgs = torch.zeros(strokes.size(0), strokes.size(1), 3, self.width, self.width)
        alphas = torch.zeros(strokes.size(0), strokes.size(1), 1, self.width, self.width)
        strokes = torch.cat([strokes, torch.ones(strokes.size(0), strokes.size(1), 1).to(strokes.device)], dim=2)
        num_control_points = [2] * self.path_num
        for b in range(strokes.size(0)):
            for num in range(strokes.size(1)):
                shapes = []
                groups = []
                shapes.append(
                    pydiffvg.Path(
                        num_control_points=torch.LongTensor(num_control_points),
                        points=strokes[b][num][:-4].reshape(-1, 2) * self.width,
                        stroke_width=torch.tensor(0.0),
                        is_closed=True))
                groups.append(
                    pydiffvg.ShapeGroup(
                        shape_ids=torch.LongTensor([num]),
                        fill_color=strokes[b][num][-4:]))
                scene_args = pydiffvg.RenderFunction.serialize_scene(self.width, self.width, shapes, groups)
                _render = pydiffvg.RenderFunction.apply
                img = _render(self.width, self.width, 2, 2, 0, None, *scene_args).permute(2, 0, 1)
                imgs[b, num] = img
        color = torch.ones_like(strokes[0][0][-4:])
        for b in range(strokes.size(0)):
            for num in range(strokes.size(1)):
                shapes = []
                groups = []
                shapes.append(
                    pydiffvg.Path(
                        num_control_points=torch.LongTensor(num_control_points),
                        points=strokes[b][num][:-4].reshape(-1, 2) * self.width,
                        stroke_width=torch.tensor(0.0),
                        is_closed=True))
                groups.append(
                    pydiffvg.ShapeGroup(
                        shape_ids=torch.LongTensor([num]),
                        fill_color=color))
                scene_args = pydiffvg.RenderFunction.serialize_scene(self.width, self.width, shapes, groups)
                _render = pydiffvg.RenderFunction.apply
                img = _render(self.width, self.width, 2, 2, 0, None, *scene_args).permute(2, 0, 1)
                alphas[b, num] = img[:,:,0,:,:]
        imgs=imgs[:,:,3,:,:]
        alpha=alphas[:,:,3,:,:]
        stroke_num=imgs.size(1)
        batch_size=imgs.size(0)
        stroke_num_map = torch.arange(1, stroke_num + 1).reshape(1, stroke_num, 1, 1, 1).repeat(batch_size, 1,1, self.width,self.width).to(self.device)
        stroke_num_map_draw = stroke_num_map * (alpha > 0.1)
        topk=1
        stroke_num_map_draw_topk, stroke_num_map_draw_topk_indices = stroke_num_map_draw.topk(topk, dim=1)
        color_stroke_topk = imgs.gather(dim=1, index=stroke_num_map_draw_topk_indices.repeat(1, 1, 3, 1, 1))
        alpha = alpha.gather(dim=1, index=stroke_num_map_draw_topk_indices)

        canvas = torch.zeros(batch_size, 3, self.width, self.width).to(self.device)

        for i in range(color_stroke_topk.shape[1]):
            canvas = canvas * (1 - alpha[:, topk - i]) + alpha[:, topk - i] * color_stroke_topk[:, topk - i]
    def loss(self, x,mask=None, require_lpips=False,epoch_id=0):
        if require_lpips and self.loss_fn_vgg is None:
            self.loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
        if self.control_num:
            # random_num_strokes=torch.randint(1,128,(int(x.size(0)),)).to(x.device)
            # random_num_strokes=random.randint(1,64)
            print('ggg')
            random_num_strokes =128
            strokes = self.encoder(x*mask,random_num_strokes)
        else:
            strokes = self.encoder(x*mask-(1-mask))
        #strokes=strokes[:,64:,:]
        pred = self.rendering(strokes)[:, :3, :, :]
        #pred.reshape(x.shape[0], -1, pred.shape[-2], pred.shape[-1])
        if mask is None:
            loss_mse = F.mse_loss(pred, x)
            return loss_mse
        else:
            #loss_mse = F.mse_loss(pred, x)
            loss_mse=F.mse_loss(pred*mask, x*mask)/(mask.sum()/(mask.size(0)*mask.size(-1)*mask.size(-2)))
            #print(loss_mse.shape,mask.sum(),mask.size(0)*mask.size(-1)*mask.size(-2))
            new_strokes=torch.cat([strokes[:,:,:-3],torch.ones_like(strokes[:,:,-3:].to(strokes.device))],dim=-1)
            pred = self.rendering(new_strokes)[:, :1, :, :]
            pred.reshape(x.shape[0], -1, pred.shape[-2], pred.shape[-1])
            # lambda_mask=max(0.1-0.001*epoch_id,0.05)
            lambda_mask = max(0.1 - 0.001 * (epoch_id+13), 0.05)
            mask = mophology.dilation(mask,m=2)
            loss_mask=((pred*(1-mask)).sum())/((1-mask).sum())*lambda_mask
            return loss_mse,loss_mask


    def loss_parallel(self, x):
        # torch.multiprocessing.set_start_method("spawn")
        strokes = self.encoder(x)
        process_list = []
        thread_num = 8
        queue = Queue()
        data_per_thread = int(x.size(0) // thread_num)
        for i in range(thread_num):
            p = Process(target=self.rendering,
                        args=(strokes[data_per_thread * i:data_per_thread * (i + 1)], None, queue, i))
            p.start()
            process_list.append(p)
        for p in process_list:
            p.join()
        results = [[] for i in range(thread_num)]
        while not queue.empty():
            i, result = queue.get()
            print(i, result.shape)
            results[i] = result
        pred = torch.cat(results, dim=0)

        pred.reshape(x.shape[0], -1, pred.shape[-2], pred.shape[-1])

        loss = F.mse_loss(pred, x)
        return loss

    # def forward_nstroke(self, x, n=32, s=0):
    #     strokes = self.encoder(x)[:, s:n].reshape(-1, 13)
    #     pred = self.rendering(strokes, batch_size=x.shape[0])
    #     pred.reshape(x.shape[0], -1, pred.shape[-2], pred.shape[-1])
    #
    #     return pred, TF.normalize(TF.resize(x, (pred.shape[-2], pred.shape[-1])), mean=MEAN, std=STD)


class AttnPainterOil(nn.Module):
    def __init__(self, stroke_num=256, stroke_dim=8, width=128):
        super(AttnPainterOil, self).__init__()
        self.encoder = StrokeAttentionPredictor(stroke_num=stroke_num, stroke_dim=stroke_dim)
        self.stroke_num = stroke_num
        self.device = 'cuda'
        if width == 128:
            self.render = FCNOil(5, True, True)
        self.width = width
        for p in self.render.parameters():
            p.requires_grad = False

    def forward(self, x):
        strokes = self.encoder(x)[:, :, :8].reshape(-1, 8)

        pred = self.rendering(strokes, batch_size=x.shape[0])
        pred.reshape(x.shape[0], -1, pred.shape[-2], pred.shape[-1])

        return pred, TF.normalize(TF.resize(x, (pred.shape[-2], pred.shape[-1])), mean=MEAN, std=STD)

    def real_forward(self, x):
        strokes = self.encoder(x)[:, :, :8].reshape(-1, 8)

        pred = self.real_rendering(strokes, batch_size=x.shape[0])
        pred.reshape(x.shape[0], -1, pred.shape[-2], pred.shape[-1])

        return pred, TF.normalize(TF.resize(x, (pred.shape[-2], pred.shape[-1])), mean=MEAN, std=STD)

    def rendering(self, strokes, batch_size):

        color_stroke, alpha, edge, stroke = self.render(strokes)
        color_stroke = color_stroke.reshape(batch_size, -1, 3, self.width, self.width)
        alpha = alpha.reshape(batch_size, -1, 1, self.width, self.width)
        stroke = stroke.reshape(batch_size, -1, 1, self.width, self.width)

        stroke_num_map = torch.arange(1, self.stroke_num + 1).reshape(1, self.stroke_num, 1, 1, 1).repeat(batch_size, 1,
                                                                                                          1, 128,
                                                                                                          128).to(
            self.device)

        stroke_num_map_draw = stroke_num_map * (alpha > 0.1)
        stroke_num_map_draw_topk, stroke_num_map_draw_topk_indices = stroke_num_map_draw.topk(10, dim=1)
        color_stroke_topk = color_stroke.gather(dim=1, index=stroke_num_map_draw_topk_indices.repeat(1, 1, 3, 1, 1))
        alpha = alpha.gather(dim=1, index=stroke_num_map_draw_topk_indices)

        canvas = torch.ones(batch_size, 3, self.width, self.width).to(self.device)

        for i in range(color_stroke_topk.shape[1]):
            canvas = canvas * (1 - alpha[:, 9 - i]) + alpha[:, 9 - i] * color_stroke_topk[:, 9 - i]

        return canvas

    def real_rendering(self, strokes, batch_size):

        color_stroke, alpha, edge, stroke = self.render.real_forward(strokes)
        color_stroke = color_stroke.reshape(batch_size, -1, 3, self.width, self.width)
        alpha = alpha.reshape(batch_size, -1, 1, self.width, self.width)
        stroke = stroke.reshape(batch_size, -1, 1, self.width, self.width)
        stroke_num_map = torch.arange(1, self.stroke_num + 1).reshape(1, self.stroke_num, 1, 1, 1).repeat(batch_size, 1,
                                                                                                          1, 128,
                                                                                                          128).to(
            self.device)
        stroke_num_map_draw = stroke_num_map * (alpha > 0.1)
        stroke_num_map_draw_topk, stroke_num_map_draw_topk_indices = stroke_num_map_draw.topk(10, dim=1)

        color_stroke_topk = color_stroke.gather(dim=1, index=stroke_num_map_draw_topk_indices.repeat(1, 1, 3, 1, 1))
        alpha = alpha.gather(dim=1, index=stroke_num_map_draw_topk_indices)

        canvas = torch.ones(batch_size, 3, self.width, self.width).to(self.device)

        for i in range(color_stroke_topk.shape[1]):
            canvas = canvas * (1 - alpha[:, 9 - i]) + alpha[:, 9 - i] * color_stroke_topk[:, 9 - i]

        return canvas

    def loss(self, x):
        strokes = self.encoder(x)[:, :, :8].reshape(-1, 8)
        pred = self.rendering(strokes, batch_size=x.shape[0])
        pred.reshape(x.shape[0], -1, pred.shape[-2], pred.shape[-1])
        loss = F.mse_loss(pred,
                          TF.normalize(TF.resize(x, (pred.shape[-2], pred.shape[-1])), mean=MEAN, std=STD).detach())
        return loss

    def forward_nstroke(self, x, n=32, s=0):
        strokes = self.encoder(x)[:, s:n].reshape(-1, 13)
        pred = self.rendering(strokes, batch_size=x.shape[0])
        pred.reshape(x.shape[0], -1, pred.shape[-2], pred.shape[-1])

        return pred, TF.normalize(TF.resize(x, (pred.shape[-2], pred.shape[-1])), mean=MEAN, std=STD)


class AttnPainter(nn.Module):
    def __init__(self, stroke_num=256, stroke_dim=13, width=128):
        super(AttnPainter, self).__init__()
        self.encoder = StrokeAttentionPredictor(stroke_num=stroke_num, stroke_dim=stroke_dim)
        self.stroke_num = stroke_num
        self.device = 'cuda'
        if width == 128:
            self.render = FCN()

        self.width = width
        for p in self.render.parameters():
            p.requires_grad = False

    def forward(self, x):
        strokes = self.encoder(x).reshape(-1, 13)
        pred = self.rendering(strokes, batch_size=x.shape[0])
        pred.reshape(x.shape[0], -1, pred.shape[-2], pred.shape[-1])

        return pred, TF.normalize(TF.resize(x, (pred.shape[-2], pred.shape[-1])), mean=MEAN, std=STD)

    def rendering(self, strokes, batch_size):

        pred = 1 - self.render(strokes[:, :10])
        pred = pred.view(batch_size, -1, self.width, self.width, 1)
        color_pred = pred * strokes[:, -3:].view(batch_size, -1, 1, 1, 3)

        pred = pred.permute(0, 1, 4, 2, 3)
        color_pred = color_pred.permute(0, 1, 4, 2, 3)

        stroke_num_map = torch.arange(1, self.stroke_num + 1).reshape(1, self.stroke_num, 1, 1, 1).repeat(batch_size, 1,
                                                                                                          1, 128,
                                                                                                          128).to(
            self.device)
        stroke_num_map_draw = stroke_num_map * (pred > 0)
        stroke_num_map_draw_topk, stroke_num_map_draw_topk_indices = stroke_num_map_draw.topk(10, dim=1)

        color_stroke_topk = color_pred.gather(dim=1, index=stroke_num_map_draw_topk_indices.repeat(1, 1, 3, 1, 1))
        stroke_topk = pred.gather(dim=1, index=stroke_num_map_draw_topk_indices)

        canvas = torch.ones(batch_size, 3, self.width, self.width).to(self.device)

        for i in range(color_stroke_topk.shape[1]):
            canvas = canvas * (1 - stroke_topk[:, 9 - i]) + color_stroke_topk[:, 9 - i]

        return canvas

    def loss(self, x):
        strokes = self.encoder(x).reshape(-1, 13)
        pred = self.rendering(strokes, batch_size=x.shape[0])
        pred.reshape(x.shape[0], -1, pred.shape[-2], pred.shape[-1])
        loss = F.mse_loss(pred,
                          TF.normalize(TF.resize(x, (pred.shape[-2], pred.shape[-1])), mean=MEAN, std=STD).detach())
        return loss

    def forward_nstroke(self, x, n=32, s=0):
        strokes = self.encoder(x)[:, s:n].reshape(-1, 13)
        pred = self.rendering(strokes, batch_size=x.shape[0])
        pred.reshape(x.shape[0], -1, pred.shape[-2], pred.shape[-1])

        return pred, TF.normalize(TF.resize(x, (pred.shape[-2], pred.shape[-1])), mean=MEAN, std=STD)

    def density_loss(self, x, density_tensor):

        strokes = self.encoder(x).reshape(-1, 13)
        pred = self.rendering(strokes, batch_size=x.shape[0])
        pred.reshape(x.shape[0], -1, pred.shape[-2], pred.shape[-1])
        density_tensor = TF.resize(density_tensor, (pred.shape[-2], pred.shape[-1]))
        loss = (density_tensor * (pred - TF.normalize(TF.resize(x, (pred.shape[-2], pred.shape[-1])), mean=MEAN,
                                                      std=STD).detach()) ** 2).mean()
        return loss, density_tensor


class AttnPainterOilDensity(nn.Module):
    '''
    based on AttnPainterV5
    '''

    def __init__(self, stroke_num=256, stroke_dim=8, width=128):
        super(AttnPainterOilDensity, self).__init__()
        self.encoder = StrokeAttentionPredictorDen(stroke_num=stroke_num, stroke_dim=stroke_dim)
        self.stroke_num = stroke_num
        self.device = 'cuda'
        if width == 128:
            self.render = FCNOil(5, True, True)
        self.width = width
        for p in self.render.parameters():
            p.requires_grad = False

    def forward(self, img, masks):
        masks = masks.float()
        x = torch.cat([img, masks], dim=1)
        # x = img
        strokes = self.encoder(x).reshape(-1, 8)
        pred, _ = self.rendering(strokes, batch_size=img.shape[0])
        pred.reshape(img.shape[0], -1, pred.shape[-2], pred.shape[-1])

        return pred, TF.normalize(TF.resize(img, (pred.shape[-2], pred.shape[-1])), mean=MEAN, std=STD)

    def real_forward(self, img, masks, res=128):
        masks = masks.float()
        x = torch.cat([img, masks], dim=1)
        strokes = self.encoder(x)[:, :, :8].reshape(-1, 8)

        pred = self.real_rendering(strokes, batch_size=img.shape[0], res=res)
        pred.reshape(img.shape[0], -1, pred.shape[-2], pred.shape[-1])

        return pred, TF.normalize(TF.resize(img, (pred.shape[-2], pred.shape[-1])), mean=MEAN, std=STD)

    def rendering(self, strokes, batch_size):

        color_stroke, alpha, edge, _ = self.render(strokes)
        color_stroke = color_stroke.reshape(batch_size, -1, 3, self.width, self.width)
        alpha = alpha.reshape(batch_size, -1, 1, self.width, self.width)
        params = strokes.reshape(batch_size, -1, 8)

        stroke_num_map = torch.arange(1, self.stroke_num + 1).reshape(1, self.stroke_num, 1, 1, 1).repeat(batch_size, 1,
                                                                                                          1, 128,
                                                                                                          128).to(
            self.device)

        stroke_s = (params[:, :, 2] * params[:, :, 3]).reshape(batch_size, -1, 1, 1, 1).repeat(1, 1, 1, 128, 128) * (
                    alpha > 0.1)
        stroke_num_map_draw = stroke_num_map * (alpha > 0.1)
        stroke_num_map_draw_topk, stroke_num_map_draw_topk_indices = stroke_num_map_draw.topk(10, dim=1)

        color_stroke_topk = color_stroke.gather(dim=1, index=stroke_num_map_draw_topk_indices.repeat(1, 1, 3, 1, 1))
        alpha = alpha.gather(dim=1, index=stroke_num_map_draw_topk_indices)

        stroke_s_map_topk = stroke_s.gather(dim=1, index=stroke_num_map_draw_topk_indices)

        canvas = torch.ones(batch_size, 3, self.width, self.width).to(self.device)
        den_map = torch.ones(batch_size, 1, self.width, self.width).to(self.device)

        for i in range(color_stroke_topk.shape[1]):
            canvas = canvas * (1 - alpha[:, 9 - i]) + alpha[:, 9 - i] * color_stroke_topk[:, 9 - i]
            den_map = den_map * (1 - alpha[:, 9 - i]) + alpha[:, 9 - i] * stroke_s_map_topk[:, 9 - i]

        return canvas, den_map

    def rendering_2(self, strokes, batch_size):

        color_stroke, alpha, edge, _ = self.render(strokes)
        color_stroke = color_stroke.reshape(batch_size, -1, 3, self.width, self.width)
        alpha = alpha.reshape(batch_size, -1, 1, self.width, self.width)
        params = strokes.reshape(batch_size, -1, 8)

        stroke_num_map = torch.arange(1, self.stroke_num + 1).reshape(1, self.stroke_num, 1, 1, 1).repeat(batch_size, 1,
                                                                                                          1, 128,
                                                                                                          128).to(
            self.device)

        stroke_s = (params[:, :, 2] * params[:, :, 3]).reshape(batch_size, -1, 1, 1, 1).repeat(1, 1, 1, 128, 128) * (
                    alpha > 0.1)

        stroke_s_map = stroke_s * (alpha > 0.1)

        canvas = torch.ones(batch_size, 3, self.width, self.width).to(self.device)
        den_map = torch.ones(batch_size, 1, self.width, self.width).to(self.device)
        for i in range(color_stroke.shape[1]):
            canvas = canvas * (1 - alpha[:, 255 - i]) + alpha[:, 255 - i] * color_stroke[:, 255 - i]
            den_map = den_map * (1 - alpha[:, 255 - i]) + alpha[:, 255 - i] * stroke_s_map[:, 255 - i]

        return canvas, den_map

    def loss(self, img, masks):
        masks = masks.float()
        x = torch.cat([img, masks], dim=1)
        strokes = self.encoder(x).reshape(-1, 8)
        pred, _ = self.rendering(strokes, batch_size=x.shape[0])
        pred.reshape(x.shape[0], -1, pred.shape[-2], pred.shape[-1])
        masks_2 = TF.resize(masks, (pred.shape[-2], pred.shape[-1])).repeat(1, 3, 1, 1).detach()
        loss = 2 * F.mse_loss(pred * masks_2, TF.normalize(TF.resize(img, (pred.shape[-2], pred.shape[-1])), mean=MEAN,
                                                           std=STD).detach() * masks_2)
        loss += F.mse_loss(pred * (1 - masks_2),
                           TF.normalize(TF.resize(img, (pred.shape[-2], pred.shape[-1])), mean=MEAN,
                                        std=STD).detach() * (1 - masks_2))
        return loss

    def forward_nstroke(self, x, n=32, s=0):
        strokes = self.encoder(x)[:, s:n].reshape(-1, 13)
        pred = self.rendering(strokes, batch_size=x.shape[0])
        pred.reshape(x.shape[0], -1, pred.shape[-2], pred.shape[-1])

        return pred, TF.normalize(TF.resize(x, (pred.shape[-2], pred.shape[-1])), mean=MEAN, std=STD)

    def density_loss(self, img, density_tensor):

        x = torch.cat([img, density_tensor], dim=1)
        strokes = self.encoder(x).reshape(-1, 8)
        pred, density_pred = self.rendering(strokes, batch_size=x.shape[0])
        pred.reshape(x.shape[0], -1, pred.shape[-2], pred.shape[-1])
        density_pred.reshape(x.shape[0], -1, pred.shape[-2], pred.shape[-1])

        density_tensor = TF.resize(density_tensor, (pred.shape[-2], pred.shape[-1]))
        loss_l2 = F.mse_loss(pred, TF.normalize(TF.resize(img, (pred.shape[-2], pred.shape[-1])), mean=MEAN,
                                                std=STD).detach())
        loss_density = (density_tensor * density_pred).mean()
        loss = loss_l2 + 0.05 * loss_density
        loss_lpips = 0
        return loss, density_tensor, loss_l2, 0.05 * loss_density, 0.01 * loss_lpips

    def real_rendering(self, strokes, batch_size, res=128):

        color_stroke, alpha, edge, stroke = self.render.real_forward(strokes, res=res)
        color_stroke = color_stroke.reshape(batch_size, -1, 3, res, res)
        alpha = alpha.reshape(batch_size, -1, 1, res, res)
        stroke = stroke.reshape(batch_size, -1, 1, res, res)
        stroke_num_map = torch.arange(1, self.stroke_num + 1).reshape(1, self.stroke_num, 1, 1, 1).repeat(batch_size, 1,
                                                                                                          1, res,
                                                                                                          res).to(
            self.device)
        stroke_num_map_draw = stroke_num_map * (alpha > 0.1)
        stroke_num_map_draw_topk, stroke_num_map_draw_topk_indices = stroke_num_map_draw.topk(10, dim=1)
        color_stroke_topk = color_stroke.gather(dim=1, index=stroke_num_map_draw_topk_indices.repeat(1, 1, 3, 1, 1))
        alpha = alpha.gather(dim=1, index=stroke_num_map_draw_topk_indices)

        canvas = torch.ones(batch_size, 3, res, res).to(self.device)

        for i in range(color_stroke_topk.shape[1]):
            canvas = canvas * (1 - alpha[:, 9 - i]) + alpha[:, 9 - i] * color_stroke_topk[:, 9 - i]

        return canvas