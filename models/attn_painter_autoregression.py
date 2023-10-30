import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from .encoder import StrokeAttentionPredictor,StrokeAttentionPredictor_autoregression, StrokeAttentionPredictorDen,StrokeAttentionPredictor_autoregression2
from .render import FCN
# from .render_oil import FCNOil
import pydiffvg
import lpips
from torch.multiprocessing import Process, Queue
from util import SVR_render
from torchvision import transforms
from util.soft_dtw_cuda import SoftDTW
import random
from torchvision.utils import save_image
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
        self.encoder = StrokeAttentionPredictor_autoregression2(stroke_num=stroke_num, stroke_dim=path_num * 6 + 3,control_num=control_num,use_resnet=True)
        self.stroke_num = stroke_num
        self.device = 'cuda'
        self.render = SVR_render.SVGObject(size=(width, width))
        self.width = width
        self.loss_fn_vgg = None
        self.dtw= SoftDTW(use_cuda=True, gamma=0.1)

    def forward(self, x, canvas,step,**kwargs):
        strokes = self.encoder(x,canvas,step=step,**kwargs)
        if step==0:
            strokes=strokes[:,:32,:]
        pred,alpha = self.rendering(strokes,require_alpha=True,**kwargs)
        newpred=alpha*pred+(1-alpha)*canvas

        return newpred,x, pred

    def predict_path(self, x,canvas=None,step=0,num=128):
        strokes = self.encoder(x,canvas,step=step,num=num)

        return strokes

    def rendering(self, strokes, save_svg_path=None, result_queue=None, idx=None,require_alpha=False,**kwargs):
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
        canvas = torch.stack(imgs, dim=0)
        if save_svg_path is not None:
            pydiffvg.save_svg(save_svg_path, self.width, self.width, shapes, groups)
        if require_alpha:
            imgs = []
            color=torch.ones_like(strokes[0][0][-4:])
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
                            fill_color=color))
                scene_args = pydiffvg.RenderFunction.serialize_scene(self.width, self.width, shapes, groups)
                _render = pydiffvg.RenderFunction.apply
                img = _render(self.width, self.width, 2, 2, 0, None, *scene_args)
                imgs.append(img.permute(2, 0, 1))
            alpha = torch.stack(imgs, dim=0)
            return canvas[:,:3,:,:],alpha[:,:3,:,:]
        else:
            return canvas[:,:3,:,:]

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

    def fast_rendering(self, strokes, save_svg_path=None):
        imgs = torch.zeros(strokes.size(0),strokes.size(1),3,self.width,self.width)
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
                imgs[b,num]=img
        if save_svg_path is not None:
            pydiffvg.save_svg(save_svg_path, self.width, self.width, shapes, groups)
        return imgs

    def f(self,x):
        return 1/(1+torch.exp(-(x-0.005)*1000))
    def loss(self, x, critic=None,epoch_id=0):
        step=random.random()
        step=1
        if step<0.5:
            canvas=torch.zeros_like(x).to(x.device)
            strokes = self.encoder(x, canvas,step=0)
            pred = self.rendering(strokes)
            if critic is None:
                loss = F.mse_loss(pred, x)
            else:
                D_fake, D_real, gradient_penalty = critic.update(pred, x)
                loss = -critic.cal_reward(pred, x).mean()
        else:
            with torch.no_grad():
                canvas = torch.zeros_like(x).to(x.device)
                strokes = self.encoder(x,canvas,step=0)
                canvas = self.rendering(strokes).detach()
            strokes = self.encoder(x,canvas,step=1)
            pred,alpha = self.rendering(strokes,require_alpha=True)
            pred = alpha * pred + (1 - alpha) * canvas
            if critic is None:
                #loss = (pred-canvas)-(pred-x)
                #y = (pred-x)**2-(canvas-x)**2
                #loss=self.f(y)*y
                #loss=loss.mean()
                #loss=((pred-x)**2-(canvas-x)**2).mean()
                loss = F.mse_loss(pred, x)
                if epoch_id<10:
                    lambda_area=max(0.04-epoch_id*0.003,0.001)
                else:
                    lambda_area = max(0.01 - epoch_id * 0.002, 0.001)
                #lambda_area = 0.002
                loss_area = -alpha.mean() * lambda_area
                loss=loss+loss_area
                #print(y.shape,self.f(y).shape)
                #loss=y*self.f(y)
            else:
                D_fake, D_real, gradient_penalty = critic.update(pred, x)
                loss = -critic.cal_reward(pred, x).mean()
        # pred.retain_grad()
        # #print(pred.require_grad)
        # # pred.register_hook(self.extract)
        # #loss=F.mse_loss(pred, x)
        # loss.backward()
        # print(pred.grad,pred.grad.min(),pred.grad.max())
        #torch.isfinite(grad_img).all()
        return loss,loss_area,F.mse_loss(pred, x)
        return loss,F.mse_loss(canvas, x),F.mse_loss(pred, x)

    def loss_optimize(self, x, new_strokes, target_strokes, ori_strokes):  # 用于优化输入new_strokes的
        canvas = torch.zeros_like(x).to(x.device)

        pred = self.rendering(torch.cat([ori_strokes, new_strokes], dim=1), require_alpha=True)[:, :3, :, :]
        loss = self.dtw(new_strokes, target_strokes).mean() * 0.0005
        loss0 = F.mse_loss(x, pred)
        return loss, loss0, canvas, pred
    def loss_stroke(self, x, critic=None,epoch_id=None):
        # if epoch_id<10:
        #     num=random.randint(1,5)
        #     num=num*8
        # else:
        num = random.randint(min(8,epoch_id), 8)
        num = num * 8
            # num = random.randint(1, 8)
            # num = num * 8
        # num=32
        record_value = {}
        with torch.no_grad():
            canvas = torch.zeros_like(x).to(x.device)
            gt_strokes = self.encoder(x,canvas,step=0)[:,:64,:].detach()
            ori_strokes=gt_strokes[:,:num,:].detach()
            canvas = self.rendering(ori_strokes).detach()
            # print(F.mse_loss(canvas, x).item())
        pred_strokes = self.encoder(x,canvas,step=1,num=16)
        #new_stroke=torch.cat([ori_strokes,pred_strokes],dim=1)
        pred, alpha = self.rendering(pred_strokes, require_alpha=True)
        pred = alpha * pred + (1 - alpha) * canvas
        #lambda_stroke=0.0009-epoch_id*0.0001
        lambda_stroke=0
        if lambda_stroke<=0:
            loss_stroke=0
            record_value['loss_stroke'] = loss_stroke
        else:
            loss_stroke=self.dtw(pred_strokes,gt_strokes[:,num:,:]).mean()*lambda_stroke
            record_value['loss_stroke'] = loss_stroke.item()
        loss_pixel=F.mse_loss(pred,x)
        loss=loss_stroke+loss_pixel
        #loss_stroke=loss_pixel

        record_value['mse1']=F.mse_loss(pred,x).item()
        record_value['mse0'] = F.mse_loss(canvas, x).item()
        record_value['area'] = alpha.sum().item()

        return loss,record_value

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