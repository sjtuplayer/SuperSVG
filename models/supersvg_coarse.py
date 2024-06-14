import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Coarse_model
import pydiffvg
import lpips
from util import SVR_render
import random
from util import mophology
channel_mean = torch.tensor([0.485, 0.456, 0.406])
channel_std = torch.tensor([0.229, 0.224, 0.225])
pydiffvg.set_print_timing(False)
pydiffvg.set_use_gpu(True)
MEAN = [-mean / std for mean, std in zip(channel_mean, channel_std)]
STD = [1 / std for std in channel_std]
torch.multiprocessing.set_start_method('spawn', force=True)


class SuperSVG_coarse(nn.Module):
    def __init__(self, stroke_num=128, path_num=4, width=128,control_num=False,self_attn_depth=1,num_loss=False):
        super(SuperSVG_coarse, self).__init__()
        self.control_num=control_num
        self.path_num = path_num
        self.encoder = Coarse_model(stroke_num=stroke_num, stroke_dim=path_num * 6 + 3,control_num=control_num,self_attn_depth=self_attn_depth,num_loss=num_loss)
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

    def rendering(self, strokes, save_svg_path=None):
        imgs = []
        if strokes.size(-1)==27:
            strokes = torch.cat([strokes, torch.ones(strokes.size(0), strokes.size(1), 1).to(strokes.device)], dim=2)
        strokes=strokes.float()
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
        return imgs

    def loss(self, x,mask=None,epoch_id=0,num_loss=True):
        strokes = self.encoder(x*mask-(1-mask))
        pred = self.rendering(strokes)[:, :3, :, :]
        loss_mse=F.mse_loss(pred*mask, x*mask)/(mask.sum()/(mask.size(0)*mask.size(-1)*mask.size(-2)))
        if strokes.size(-1)==27:
            new_strokes=torch.cat([strokes[:,:,:-3],torch.ones_like(strokes[:,:,-3:].to(strokes.device))],dim=-1)
        else:
            new_strokes = torch.cat([strokes[:, :, :-4], torch.ones_like(strokes[:, :, -4:].to(strokes.device))],
                                    dim=-1)
        pred = self.rendering(new_strokes)[:, :1, :, :]
        pred.reshape(x.shape[0], -1, pred.shape[-2], pred.shape[-1])
        lambda_mask = max(0.05 - 0.005 * (epoch_id+1), 0.01)
        mask = mophology.dilation(mask,m=2)
        loss_mask=((pred*(1-mask)).sum())/((1-mask).sum())*lambda_mask
        loss = loss_mse + loss_mask
        log_loss={}
        log_loss['loss_pixel'] = loss_mse
        log_loss['loss_mask']=loss_mask.item()
        if num_loss and strokes.size(-1)==28:
            loss_num=strokes[:,:,-1].sum(dim=-1)
            loss_num=loss_num.mean()
            loss+=loss_num*0.00001
            log_loss['loss_num'] = loss_num*0.00001
            log_loss['path_num'] = loss_num
        log_loss["loss"] = loss.item()
        return loss,log_loss
