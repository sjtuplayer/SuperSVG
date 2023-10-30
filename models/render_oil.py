import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from .morphology import dilation
from PIL import Image
import torchvision.transforms.functional as TF


def read_img(img_path, img_type='RGB'):
    img = Image.open(img_path).convert(img_type)
    img = np.array(img)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float() / 255.
    return img
brush_large_vertical = read_img('brushes/brush_fromweb2_large_vertical.png', 'L').cuda()
brush_large_horizontal = read_img('brushes/brush_fromweb2_large_horizontal.png', 'L').cuda()
meta_brushes = torch.cat([brush_large_vertical, brush_large_horizontal], dim=0)


def normal(x, width):
    return (int)(x * (width - 1) + 0.5)

def update_transformation_matrix(M, m):
    M_ = np.concatenate([M, np.zeros([1,3])], axis=0)
    M_[-1, -1] = 1
    m_ = np.concatenate([m, np.zeros([1,3])], axis=0)
    m_[-1, -1] = 1

    M_new = np.matmul(m_, M_)
    return M_new[0:2, :]

def build_scale_matrix(sx, sy):
    transform_matrix = np.zeros((2, 3))
    transform_matrix[0, 0] = sx
    transform_matrix[1, 1] = sy
    return transform_matrix

def build_transformation_matrix(transform):
    """Convert transform list to transformation matrix

    :param transform: transform list as [dx, dy, da]
    :return: transform matrix as 2d (2, 3) numpy array
    """
    transform_matrix = np.zeros((2, 3))

    transform_matrix[0, 0] = np.cos(transform[2])
    transform_matrix[0, 1] = -np.sin(transform[2])
    transform_matrix[1, 0] = np.sin(transform[2])
    transform_matrix[1, 1] = np.cos(transform[2])
    transform_matrix[0, 2] = transform[0]
    transform_matrix[1, 2] = transform[1]

    return transform_matrix


class FCNOil(nn.Module):
    def __init__(self,d=10,need_alphas=True,need_edge=True):
        super(FCNOil, self).__init__()
        self.need_alphas = need_alphas
        self.fc1 = (nn.Linear(d, 512))
        self.fc2 = (nn.Linear(512, 1024))
        self.fc3 = (nn.Linear(1024, 2048))
        self.fc4 = (nn.Linear(2048, 4096))
        self.conv1 = (nn.Conv2d(16, 32, 3, 1, 1))
        self.conv2 = (nn.Conv2d(32, 32, 3, 1, 1))
        self.conv3 = (nn.Conv2d(8, 16, 3, 1, 1))
        self.conv4 = (nn.Conv2d(16, 16, 3, 1, 1))
        self.conv5 = (nn.Conv2d(4, 8, 3, 1, 1))
        if need_edge:
            self.conv6 = (nn.Conv2d(8, 12, 3, 1, 1))
        elif need_alphas:
            self.conv6 = (nn.Conv2d(8, 8, 3, 1, 1))
        else:
            self.conv6 = (nn.Conv2d(8, 4, 3, 1, 1))
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.stylized = False


    def forward(self,x):
        x=x.squeeze()
        tmp = 1 - self.draw(x[:, :-3])
        stroke = tmp[:, 0]
        alpha = tmp[:, 1]
        edge=tmp[:, 2]
        stroke = stroke.view(-1, 128, 128, 1)
        alpha = alpha.view(-1, 128, 128, 1)
        edge = edge.view(-1, 128, 128, 1)
        color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
        alpha = alpha.permute(0, 3, 1, 2)
        color_stroke = color_stroke.permute(0, 3, 1, 2)
        edge = edge.permute(0, 3, 1, 2)
        return color_stroke, alpha, edge, stroke
    
    def draw(self, x):
        b = x.size(0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.view(-1, 16, 16, 16)
        x = F.relu(self.conv1(x))
        x = self.pixel_shuffle(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pixel_shuffle(self.conv6(x))
        x = torch.sigmoid(x)
        return 1 - x.view(b,-1, 128, 128).squeeze()

    def real_forward(self, x, res=128):
        x=x.squeeze()
        tmp = 1 - self.draw(x[:, :-3])
        strokes, alphas = self.draw_oil(x, size=res)
        stroke = strokes[:,0] 
        alpha = TF.resize(tmp[:, 1],(res, res)).float() 
        edge = None
        stroke = stroke.reshape(-1, res, res, 1)
        alpha = alpha.reshape(-1, res, res, 1)
        color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
        alpha = alpha.permute(0, 3, 1, 2)
        color_stroke = color_stroke.permute(0, 3, 1, 2)
        return color_stroke, alpha, edge, stroke

    def draw_oil(self, param, size=128):
        H=W=size
        b = param.shape[0]
        param_list = torch.split(param, 1, dim=1)
        x0, y0, w, h, theta = [item.squeeze(-1) for item in param_list[:5]]
        sin_theta = torch.sin(torch.acos(torch.tensor(-1., device=param.device)) * theta)
        cos_theta = torch.cos(torch.acos(torch.tensor(-1., device=param.device)) * theta)
        index = torch.full((b,), -1, device=param.device)
        index[h > w] = 0
        index[h <= w] = 1
        brush = meta_brushes[index.long()]
        alphas = meta_brushes[index.long()]
        warp_00 = cos_theta / w
        warp_01 = sin_theta * H / (W * w)
        warp_02 = (1 - 2 * x0) * cos_theta / w + (1 - 2 * y0) * sin_theta * H / (W * w)
        warp_10 = -sin_theta * W / (H * h)
        warp_11 = cos_theta / h
        warp_12 = (1 - 2 * y0) * cos_theta / h - (1 - 2 * x0) * sin_theta * W / (H * h)
        warp_0 = torch.stack([warp_00, warp_01, warp_02], dim=1)
        warp_1 = torch.stack([warp_10, warp_11, warp_12], dim=1)
        warp = torch.stack([warp_0, warp_1], dim=1)
        grid = torch.nn.functional.affine_grid(warp, torch.Size((b, 3, H, W)), align_corners=False)
        brush = torch.nn.functional.grid_sample(brush, grid, align_corners=False)
        alphas = torch.nn.functional.grid_sample(alphas, grid, align_corners=False)
        brush = dilation(brush, m=1)
        
        return brush, alphas

