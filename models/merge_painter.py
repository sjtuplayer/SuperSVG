import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from .encoder import StrokeAttentionPredictor, StrokeAttentionPredictorDen, Path_Transformer
from .render import FCN
# from .render_oil import FCNOil
import pydiffvg
import lpips
from torch.multiprocessing import Process, Queue
from util import SVR_render
from torchvision import transforms
import time
channel_mean = torch.tensor([0.485, 0.456, 0.406])
channel_std = torch.tensor([0.229, 0.224, 0.225])
pydiffvg.set_print_timing(False)
pydiffvg.set_use_gpu(True)
MEAN = [-mean / std for mean, std in zip(channel_mean, channel_std)]
STD = [1 / std for std in channel_std]
torch.multiprocessing.set_start_method('spawn', force=True)


class Merge_model(nn.Module):
    def __init__(self, stroke_num=128, path_num=4, width=128):
        super(Merge_model, self).__init__()
        self.path_num = path_num
        self.encoder = Path_Transformer(stroke_num=stroke_num, stroke_dim=path_num * 6 + 3)
        self.stroke_num = stroke_num
        self.device = 'cuda'
        self.render = SVR_render.SVGObject(size=(width, width))
        self.width = width
        self.loss_fn_vgg = None
        #self.resize=transforms.Resize()

    def forward(self, paths,images, **kwargs):
        strokes = self.encoder(paths, images)
        pred = self.rendering(strokes)[:, :3, :, :]
        return pred, images

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

    def loss(self, paths, images):
        strokes = self.encoder(paths,images)
        pred = self.rendering(strokes)[:, :3, :, :]
        loss = F.mse_loss(pred, images)
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
