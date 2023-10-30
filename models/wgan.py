import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam, SGD
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import grad as torch_grad
import torch.nn.utils.weight_norm as weightNorm
import random
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torchvision.utils import save_image
from torchvision import transforms
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dim = 128
LAMBDA = 10 # Gradient penalty lambda hyperparameter

class TReLU(nn.Module):
    def __init__(self):
            super(TReLU, self).__init__()
            self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.alpha.data.fill_(0)

    def forward(self, x):
        x = F.relu(x - self.alpha) + self.alpha
        return x

class Discriminator(nn.Module):
        def __init__(self,input_dim=6):
            super(Discriminator, self).__init__()
            self.conv0 = weightNorm(nn.Conv2d(input_dim, 16, 5, 2, 2))
            self.conv1 = weightNorm(nn.Conv2d(16, 32, 5, 2, 2))
            self.conv2 = weightNorm(nn.Conv2d(32, 64, 5, 2, 2))
            self.conv3 = weightNorm(nn.Conv2d(64, 128, 5, 2, 2))
            self.conv4 = weightNorm(nn.Conv2d(128, 1, 5, 2, 2))
            self.relu0 = TReLU()
            self.relu1 = TReLU()
            self.relu2 = TReLU()
            self.relu3 = TReLU()

        def forward(self, x):
            x = self.conv0(x)
            x = self.relu0(x)
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.conv3(x)
            x = self.relu3(x)
            x = self.conv4(x)
            x = F.avg_pool2d(x, 4)
            x = x.view(-1, 1)
            return x
class Wgan:
    def __init__(self,opts,input_dim=6,dataset_inside=False):
        self.opts=opts
        self.dataset_inside=dataset_inside
        self.input_dim=input_dim
        self.net=Discriminator(input_dim).cuda()
        self.optimizerD = Adam(self.net.parameters(), lr=3e-4, betas=(0.5, 0.999))
        if dataset_inside:
            self.create_dataset()
    def load(self,path):
        self.net.load_state_dict(torch.load(path))
    def create_dataset(self):
        class Train_Data(Dataset):
            def __init__(self, img_path):
                self.loader = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize([128, 128])
                ])
                self.data_path = img_path
                dirs = os.listdir(self.data_path)
                self.file_names = []
                if 'imagenet' in img_path:
                    for dir in dirs:
                        dirs2 = os.listdir(os.path.join(self.data_path, dir))
                        for name in dirs2:
                            self.file_names.append(os.path.join(dir, name))
                if 'celeba' in img_path:
                    for name in dirs:
                        self.file_names.append(os.path.join(self.data_path, name))
                self.l = len(self.file_names)
                print(self.l)
            def __getitem__(self, idx):
                image = Image.open(os.path.join(self.data_path, self.file_names[idx])).convert('RGB')
                image = self.loader(image)
                return image
            def __len__(self):
                return self.l

        self.train_data = Train_Data('/home/huteng/dataset/celeba')
        self.train_dataloader = DataLoader(self.train_data,
                                           batch_size=self.opts.batch_size,
                                           shuffle=True,
                                           num_workers=8,
                                           drop_last=True)
        self.train_iter = iter(self.train_dataloader)
        self.iter = iter(self.train_dataloader)
    def get_tar_img(self):
        x=next(self.iter,None)
        if x is None:
            print('reload from dataset')
            self.iter = iter(self.train_dataloader)
            return next(self.iter).cuda()
        else:
            return x.cuda()
    def cal_gradient_penalty(self, real_data, fake_data, batch_size):
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous()
        alpha = alpha.view(batch_size, -1, dim, dim)
        alpha = alpha.to(device)
        fake_data = fake_data.view(batch_size, -1, dim, dim)
        interpolates = Variable(alpha * real_data.data + ((1 - alpha) * fake_data.data), requires_grad=True)
        disc_interpolates = self.net(interpolates)
        gradients = autograd.grad(disc_interpolates, interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                  create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
        return gradient_penalty

    def cal_reward(self,fake_data, real_data):
        if self.input_dim==6:
            return self.net(torch.cat([real_data, fake_data], 1))
        else:
            return self.net(fake_data)

    def update(self,fake_data, real_data,blur_data=None):
        if self.input_dim==6:
            if blur_data is None:
                fake_data = fake_data.detach()
                real_data = real_data.detach()
                fake = torch.cat([real_data, fake_data], 1)
                real = torch.cat([real_data, real_data], 1)
            else:
                fake_data = fake_data.detach()
                real_data = real_data.detach()
                blur_data=  blur_data.detach()
                fake = torch.cat([real_data, fake_data], 1)
                real = torch.cat([real_data, blur_data], 1)
        else:
            if self.dataset_inside:
                fake = fake_data.detach()
                real = self.get_tar_img()
            else:
                fake = fake_data.detach()
                real = real_data.detach()
        D_real = self.net(real)
        D_fake = self.net(fake)
        gradient_penalty = self.cal_gradient_penalty( real, fake, real.shape[0])
        self.optimizerD.zero_grad()
        D_cost = D_fake.mean() - D_real.mean() + gradient_penalty
        D_cost.backward()
        self.optimizerD.step()
        return D_fake.mean(), D_real.mean(), gradient_penalty

class Discriminator2(nn.Module):
    def __init__(self, input_dim=6):
        super(Discriminator2, self).__init__()
        self.conv0 = weightNorm(nn.Conv2d(input_dim, 16, 5, 2, 2))
        self.conv1 = weightNorm(nn.Conv2d(16, 32, 5, 2, 2))
        self.conv2 = weightNorm(nn.Conv2d(32, 64, 5, 2, 2))
        self.conv3 = weightNorm(nn.Conv2d(64, 128, 5, 2, 2))
        self.conv4 = weightNorm(nn.Conv2d(128, 256, 5, 2, 2))
        self.conv5 = weightNorm(nn.Conv2d(256, 256, 5, 2, 2))
        self.conv6 = weightNorm(nn.Conv2d(256, 1, 5, 2, 2))
        self.relu0 = TReLU()
        self.relu1 = TReLU()
        self.relu2 = TReLU()
        self.relu3 = TReLU()
        self.relu4 = TReLU()
        self.relu5 = TReLU()

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu0(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x=self.relu4(self.conv5(x))
        x=self.relu5(self.conv6(x))
        x = F.avg_pool2d(x, 4)
        x = x.view(-1, 1)
        return x


class Wgan2:
    def __init__(self, input_dim=3, action=False):
        self.action = action
        self.input_dim = input_dim
        self.net = Discriminator2(input_dim).cuda()
        self.optimizerD = Adam(self.net.parameters(), lr=3e-4, betas=(0.5, 0.999))

    def load(self, path):
        self.net.load_state_dict(torch.load(path))


    def cal_gradient_penalty(self, real_data, fake_data, batch_size):
        size=real_data.size(-2)
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand(batch_size, int(real_data.nelement() / batch_size)).contiguous()
        alpha = alpha.view(batch_size, -1, size, size)
        alpha = alpha.to(device)
        fake_data = fake_data.view(batch_size, -1, size, size)
        interpolates = Variable(alpha * real_data.data + ((1 - alpha) * fake_data.data), requires_grad=True)
        disc_interpolates = self.net(interpolates)
        gradients = autograd.grad(disc_interpolates, interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                  create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
        return gradient_penalty

    def cal_reward(self, fake_data):
        if self.input_dim == 6:
            return self.net(fake_data)
        else:
            return self.net(fake_data)
    def update(self, fake_data, real_data):
        fake = fake_data.detach()
        real = real_data.detach()
        D_real = self.net(real)
        D_fake = self.net(fake)
        gradient_penalty = self.cal_gradient_penalty(real, fake, real.shape[0])
        self.optimizerD.zero_grad()
        D_cost = D_fake.mean() - D_real.mean() + gradient_penalty
        D_cost.backward()
        self.optimizerD.step()
        return D_fake.mean(), D_real.mean(), gradient_penalty
class Contrastive_net(nn.Module):
    def __init__(self,input_dim=6):
        super(Contrastive_net, self).__init__()
        self.conv0 = weightNorm(nn.Conv2d(input_dim, 16, 5, 2, 2))
        self.conv1 = weightNorm(nn.Conv2d(16, 32, 5, 2, 2))
        self.conv2 = weightNorm(nn.Conv2d(32, 64, 5, 2, 2))
        self.conv3 = weightNorm(nn.Conv2d(64, 128, 5, 2, 2))
        self.conv4 = weightNorm(nn.Conv2d(128, 128, 5, 2, 2))
        self.relu0 = TReLU()
        self.relu1 = TReLU()
        self.relu2 = TReLU()
        self.relu3 = TReLU()
        self.linear_param = nn.Sequential(
            nn.Linear(128*4*4, 128)
        )
    def forward(self, x):
        x = self.conv0(x)
        x = self.relu0(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x=self.linear_param(x.flatten(1))
        return x
class Contrastive:
    def __init__(self,input_dim=6):
        self.input_dim=input_dim
        self.net=Contrastive_net(input_dim).cuda()
        self.optimizerD = Adam(self.net.parameters(), lr=3e-4, betas=(0.5, 0.999))
    def cal_reward(self,fake_data, real_data):
        if self.input_dim==6:
            fake = torch.cat([real_data, fake_data], 1)
            real = torch.cat([real_data, real_data], 1)
            D_real = self.net(real)
            D_fake = self.net(fake)
            D_cost = torch.cosine_similarity(D_real, D_fake, dim=1)
            return D_cost
        else:
            return self.net(fake_data)
    # def infoNCE_loss(self,x1,x2):
    #     return torch.cosine_similarity(x1,x2,dim=1)
    def update(self,fake_data, real_data,blur_data=None):
        if self.input_dim==6:
            if blur_data is None:
                fake_data = fake_data.detach()
                real_data = real_data.detach()
                fake = torch.cat([real_data, fake_data], 1)
                real = torch.cat([real_data, real_data], 1)
            else:
                fake_data = fake_data.detach()
                real_data = real_data.detach()
                blur_data=  blur_data.detach()
                fake = torch.cat([real_data, fake_data], 1)
                real = torch.cat([real_data, blur_data], 1)
        else:
            fake = fake_data.detach()
            real = real_data.detach()
        D_real = self.net(real)
        D_fake = self.net(fake)
        self.optimizerD.zero_grad()
        D_cost = torch.cosine_similarity(D_real, D_fake,dim=1).mean()
        D_cost.backward()
        self.optimizerD.step()
        return D_cost,torch.tensor(0),torch.tensor(0)
