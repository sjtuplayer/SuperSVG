import pydiffvg
import torch
import cv2
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import warnings

warnings.filterwarnings("ignore")

import os
import copy
import lpips
# import skfmm
#from .xing_loss import xing_loss

import time
from tqdm import tqdm

pydiffvg.set_print_timing(False)
pydiffvg.set_use_gpu(True)
gamma = 1.0


##########
# helper #
##########

class SVGObject:
    def __init__(self, size=None, init_paths=None, init_colors=None, svg_path=None):
        assert svg_path is not None or size is not None
        if svg_path is not None:
            canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(svg_path)
            self.width, self.height = canvas_width, canvas_height
            self.layers = [{'shapes': shapes, 'shape_groups': shape_groups}]
            self.shape_id = len(shapes)
        else:
            self.height, self.width = size
            self.layers = []
            self.shape_id = 0
            assert init_paths is None or init_colors is None or len(init_paths) == len(init_colors)
            if init_paths is not None and init_colors is not None:
                self.add_paths(init_paths, init_colors)

        # for finetuning
        self.target = None
        self.optim_schedular_dict = {}
        self.loss_weight_keep = 0.0

    # new layer every call
    def add_paths(self, paths, colors):

        # new layer
        layer = {'shapes': [], 'shape_groups': []}
        self.layers.append(layer)

        for path, color in zip(paths, colors):
            path = torch.FloatTensor(path)

            num_control_points = [2] * int(path.shape[0] / 3)

            self.layers[-1]['shapes'].append(
                pydiffvg.Path(
                    num_control_points=torch.LongTensor(num_control_points),
                    points=path,
                    stroke_width=torch.tensor(0.0),
                    is_closed=True))

            fill_color = torch.FloatTensor(color) / 255

            self.layers[-1]['shape_groups'].append(
                pydiffvg.ShapeGroup(
                    shape_ids=torch.LongTensor([self.shape_id]),
                    fill_color=fill_color))

            self.shape_id += 1

    def get_trainable_var(self, layer_id=-1):
        point_var = []
        color_var = {}

        for path in self.layers[layer_id]['shapes']:
            path.points.requires_grad = True
            point_var.append(path.points)
        for group in self.layers[layer_id]['shape_groups']:
            group.fill_color.requires_grad = True
            # color_var.append(group.fill_color)
            color_var[group.fill_color.data_ptr()] = group.fill_color
        color_var = list(color_var.values())

        return point_var, color_var

    def get_all_shapes(self):
        shapes = []
        shape_groups = []
        for layer in self.layers:
            shapes += layer['shapes']
            shape_groups += layer['shape_groups']

        return shapes, shape_groups

    def render(self, shapes=None, shape_groups=None, for_sdf=False):
        if shapes is None or shape_groups is None:
            shapes, shape_groups = self.get_all_shapes()
        scene_args = pydiffvg.RenderFunction.serialize_scene(self.width, self.height, shapes, shape_groups)
        _render = pydiffvg.RenderFunction.apply
        img = _render(self.width, self.height, 2, 2, 0, None, *scene_args)
        # Compose img with white background
        para_bg = torch.tensor([1., 1., 1.], requires_grad=False, device=img.device)
        if not for_sdf:
            img = img[:, :, 3:4] * img[:, :, :3] + para_bg * (1 - img[:, :, 3:4])
        else:
            img = img[:, :, 3]

        return img

    def save(self, filename):
        shapes, shape_groups = self.get_all_shapes()
        pydiffvg.save_svg(filename, self.width, self.height, shapes, shape_groups)

    def set_target(self, target):
        assert target.shape == (self.height, self.width, 3), "target shape should be (height, width, 3)"
        target = torch.FloatTensor((target / 255).astype(np.float32)).permute(2, 0, 1)[None].to(pydiffvg.get_device())
        self.target = target

    def finetune(self, lr_path=1.0, lr_color=0.01, num_iter=5, decay_every=1, decay_ratio=0.2, loss_type='mse',
                 x_loss_weight=0.01, use_alpha=True, save_options=None, loss_weight_map=None):
        '''
        finetune the svg object to match the target image. ***Multiple calls is not tested!

        Args:
            lr_path: learning rate for path points
            lr_color: learning rate for color
            num_iter: number of iterations
            decay_every: decay learning rate every decay_every epoch, calling this function once is one epoch
            decay_ratio: decay learning rate by decay_ratio
            loss_type: 'mse' or 'sdf' or 'lpips'
            x_loss_weight: weight for xing loss
            use_alpha: update alpha channel of paths
            save_options: dict, save intermediate results, e.g. {'path': 'path/to/save', 'iters': [1, 2, 3]}
        '''
        assert self.target is not None, "target is not set"
        assert len(self.layers) > 0, "no layers to finetune"
        device = self.target.device

        lrlambda_f = linear_decay_lrlambda_f(decay_every, decay_ratio)
        new_layers = [len(self.optim_schedular_dict), len(self.layers)]

        for layer_id in range(*new_layers):
            point_var, color_var = self.get_trainable_var(layer_id)
            optimizer = torch.optim.Adam([{'params': point_var, 'lr': lr_path},
                                          {'params': color_var, 'lr': lr_color}])
            scheduler = LambdaLR(optimizer, lr_lambda=lrlambda_f)
            self.optim_schedular_dict[layer_id] = (optimizer, scheduler)

        shapes, shape_groups = self.get_all_shapes()

        # point_var for xing loss
        point_var_all = []
        for layer in self.layers:
            for path in layer['shapes']:
                point_var_all.append(path.points)

        loss_list = []

        start_time = time.time()

        if loss_type == 'lpips':
            loss_fn_lpips = lpips.LPIPS(net='alex').to(device)

        pbar = tqdm(range(num_iter))
        for i in pbar:
            for _, (optim, _) in self.optim_schedular_dict.items():
                optim.zero_grad()

            # render SVG
            img = self.render(shapes, shape_groups)
            x = img.unsqueeze(0).permute(0, 3, 1, 2)  # HWC -> NCHW

            # save image for animation
            if save_options is not None and 'save_gif' in save_options:
                save_path = save_options['save_gif'] + '/iter_{}.png'.format(i)
                imshow = img.detach().cpu()
                pydiffvg.imwrite(imshow, save_path, gamma=1.0)

            # loss
            loss = ((x - self.target) ** 2)

            loss_weight = None
            if loss_type == 'sdf':
                shapes_forsdf = copy.deepcopy(shapes)
                shape_groups_forsdf = copy.deepcopy(shape_groups)
                for si in shapes_forsdf:
                    si.stroke_width = torch.FloatTensor([0]).to(device)
                for sg_idx, sgi in enumerate(shape_groups_forsdf):
                    sgi.fill_color = torch.FloatTensor([1, 1, 1, 1]).to(device)
                    sgi.shape_ids = torch.LongTensor([sg_idx]).to(device)

                with torch.no_grad():
                    im_forsdf = self.render(shapes_forsdf, shape_groups_forsdf, for_sdf=True)
                # use alpha channel is a trick to get 0-1 image
                im_forsdf = (im_forsdf).detach().cpu().numpy()
                loss_weight = self.get_sdf(im_forsdf, normalize='to1')
                loss_weight += self.loss_weight_keep
                loss_weight = np.clip(loss_weight, 0, 1)
                loss_weight = torch.FloatTensor(loss_weight).to(device)

            if loss_type == 'mse' or loss_type == 'lpips':
                if loss_weight_map is not None:
                    loss_weight = torch.FloatTensor(loss_weight_map).to(device)
                    loss = (loss.sum(1) * loss_weight).mean()
                else:
                    loss = loss.sum(1).mean()

            elif loss_type == 'sdf':
                loss = (loss.sum(1) * loss_weight).mean()

            # if x_loss_weight > 0:
            #     loss_xing = xing_loss(point_var_all)
            #     loss += loss_xing * x_loss_weight

            if loss_type == 'lpips':
                loss_lpips = loss_fn_lpips(x, self.target).reshape(loss.shape)
                loss += loss_lpips

            loss_list.append(loss.item())
            loss.backward()

            if not use_alpha:
                for group in shape_groups:
                    group.fill_color.grad[3] = 0

            pbar.set_description(f"loss: {loss_list[-1] : .5f}")

            # optimize
            for _, (optim, _) in self.optim_schedular_dict.items():
                optim.step()

            for group in shape_groups:
                group.fill_color.data.clamp_(0.0, 1.0)

            if save_options is not None:
                if i in save_options['save_iters']:
                    self.save(os.path.join(save_options['save_dir'], f'{i}.svg'))

        for _, (_, sched) in self.optim_schedular_dict.items():
            sched.step()

        end_time = time.time()
        print(f"finetune time: {end_time - start_time}, loss: {loss_list[-1]}")

        if loss_type == 'sdf':
            self.loss_weight_keep = loss_weight.detach().cpu().numpy()

        return loss_list[-1]

    def get_sdf(self, phi, method='skfmm', **kwargs):
        if method == 'skfmm':
            import skfmm
            phi = (phi - 0.5) * 2
            if (phi.max() <= 0) or (phi.min() >= 0):
                return np.zeros(phi.shape).astype(np.float32)
            sd = skfmm.distance(phi, dx=1)

            flip_negative = kwargs.get('flip_negative', True)
            if flip_negative:
                sd = np.abs(sd)

            truncate = kwargs.get('truncate', 10)
            sd = np.clip(sd, -truncate, truncate)
            # print(f"max sd value is: {sd.max()}")

            zero2max = kwargs.get('zero2max', True)
            if zero2max and flip_negative:
                sd = sd.max() - sd
            elif zero2max:
                raise ValueError

            normalize = kwargs.get('normalize', 'sum')
            if normalize == 'sum':
                sd /= sd.sum()
            elif normalize == 'to1':
                sd /= sd.max()
            return sd


class linear_decay_lrlambda_f(object):
    def __init__(self, decay_every, decay_ratio):
        self.decay_every = decay_every
        self.decay_ratio = decay_ratio

    def __call__(self, n):
        decay_time = n // self.decay_every
        decay_step = n % self.decay_every
        lr_s = self.decay_ratio ** decay_time
        lr_e = self.decay_ratio ** (decay_time + 1)
        r = decay_step / self.decay_every
        lr = lr_s * (1 - r) + lr_e * r
        return lr