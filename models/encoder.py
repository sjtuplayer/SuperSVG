import torch
import torch.nn as nn
from collections import OrderedDict
import timm 
from timm.models.vision_transformer import Block
from util.cross_attention import CrossAttentionBlock
from torchvision import transforms
from torchvision import models
from util.utils import SignWithSigmoidGrad

class StrokeAttentionHead(nn.Module):
    '''
    Cross-Attention-based Head
    '''
    def __init__(self, stroke_num=256, stroke_dim=13, encoder_embed_dim=768, self_attn_depth=4,control_num=False,num_loss=False):
        super(StrokeAttentionHead, self).__init__()
        self.control_num=control_num
        if num_loss:
            stroke_dim+=1
        self.stroke_dim=stroke_dim
        if control_num:
            self.control_num_tokens = nn.Parameter(torch.zeros(1, 8, stroke_num))
        self.stroke_tokens = nn.Parameter(torch.zeros(1, stroke_dim, stroke_num))
        self.cross_attn_block = CrossAttentionBlock(x_dim=encoder_embed_dim, y_dim=stroke_num, num_heads=8)
        self.self_attn_blocks = nn.Sequential(*[
            Block(
                dim=stroke_num, num_heads=8, mlp_ratio=4., qkv_bias=True,
                attn_drop=0., norm_layer=nn.LayerNorm, act_layer=nn.GELU)
            for i in range(self_attn_depth)])
        self.linear_head = nn.Linear(stroke_dim, stroke_dim)
    def forward(self, x,num=None):
        if self.control_num:
            assert num is not None, 'num should not be None if control_num is True'
            control_num_tokens = self.control_num_tokens.repeat(x.shape[0],1,1)*num
            x=self.cross_attn_block(x, torch.cat([self.stroke_tokens.repeat(x.shape[0],1,1),control_num_tokens],dim=1))[:,:self.stroke_dim,:]
            x = self.self_attn_blocks(x)
            x = self.linear_head(x.permute(0, 2, 1))[:, :num, :]
        else:
            x = self.cross_attn_block(x, self.stroke_tokens.repeat(x.shape[0],1,1))
            x = self.self_attn_blocks(x)
            x = self.linear_head(x.permute(0, 2, 1))
        return x
    
class Coarse_model(nn.Module):
    '''
    Cross-Attention-based Predictor
    '''
    def __init__(self, stroke_num=512, stroke_dim=13, self_attn_depth=1,control_num=False,num_loss=False):
        super(Coarse_model, self).__init__()
        self.feature_extractor = timm.create_model('vit_small_patch16_224_dino', pretrained=False)
        self.stroke_head = StrokeAttentionHead(stroke_num=stroke_num, stroke_dim=stroke_dim, encoder_embed_dim=self.feature_extractor.embed_dim, self_attn_depth=self_attn_depth,control_num=control_num,num_loss=num_loss)
        self.resize_224 = transforms.Resize((224, 224))
    def extract_features(self, x):
        x = self.feature_extractor.patch_embed(x)
        cls_token = self.feature_extractor.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.feature_extractor.pos_drop(x + self.feature_extractor.pos_embed)
        x = self.feature_extractor.blocks(x)
        x = self.feature_extractor.norm(x)
        return x[:,1:]

    def forward(self, x,num=None,**kwargs):
        if x.size(-1) != 224:
            x = self.resize_224(x)
        x = self.extract_features(x)
        x = self.stroke_head(x,num)
        x=torch.sigmoid(x)
        if x.size(-1)==28:
            x=torch.cat([x[:,:,:27],SignWithSigmoidGrad.apply(x[:,:,27:28]-0.5)],dim=-1)
        return x

class path_predictor(nn.Module):
    def __init__(self, stage,stroke_num=512, stroke_dim=13, self_attn_depth=1,control_num=False,use_resnet=False,num_loss=False):
        super(path_predictor, self).__init__()

        self.stage=stage
        self.stroke_num=stroke_num
        self.resize_224 = transforms.Resize((224, 224))
        if stage==0:
            self.feature_extractor = timm.create_model('vit_small_patch16_224_dino', pretrained=False)
            self.stroke_head = StrokeAttentionHead(stroke_num=stroke_num, stroke_dim=stroke_dim,
                                                   encoder_embed_dim=self.feature_extractor.embed_dim,
                                                   self_attn_depth=self_attn_depth, control_num=control_num,num_loss=num_loss)
        else:
            self.use_resnet=use_resnet
            self.map_in = nn.Sequential(
                nn.Conv2d(6, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 3, 3, 1, 1),
                nn.BatchNorm2d(3),
                nn.ReLU(),
            )
            if not use_resnet:
                self.feature_extractor = timm.create_model('vit_small_patch16_224_dino', pretrained=False)
                self.stroke_head = StrokeAttentionHead(stroke_num=128, stroke_dim=stroke_dim,
                                                       encoder_embed_dim=self.feature_extractor.embed_dim,
                                                       self_attn_depth=self_attn_depth, control_num=control_num)
                self.map_out=nn.Linear(128, stroke_num)
            else:
                self.resnet=models.resnet50(pretrained=True)
                num_ftrs = self.resnet.fc.in_features
                print('num featrues',num_ftrs)
                self.resnet.fc = nn.Linear(num_ftrs, stroke_num*stroke_dim)
    def extract_features(self, x):
        x = self.feature_extractor.patch_embed(x)
        cls_token = self.feature_extractor.cls_token.expand(x.shape[0], -1,-1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.feature_extractor.pos_drop(x + self.feature_extractor.pos_embed)
        x = self.feature_extractor.blocks(x)
        x = self.feature_extractor.norm(x)
        return x[:, 1:]

    def forward(self, x, canvas):
        if x.size(-1) != 224:
            x = self.resize_224(x)
            if canvas is not None:
                canvas = self.resize_224(canvas)
        if self.stage==0:
            x = self.extract_features(x)
            x = self.stroke_head(x)
        else:
            if self.use_resnet:
                x = torch.cat([x, canvas], dim=1)
                x = self.map_in(x)
                x = self.resnet(x)
                x=x.view(x.size(0),self.stroke_num,-1)
            else:
                x = torch.cat([x, canvas], dim=1)
                x = self.map_in(x)
                x = self.extract_features(x)
                x = self.stroke_head(x)
                x= self.map_out(x.permute(0,2,1)).permute(0,2,1)

        x = torch.sigmoid(x)
        if x.size(-1) == 28:
            x = torch.cat([x[:, :, :27], SignWithSigmoidGrad.apply(x[:, :, 27:28] - 0.5)], dim=-1)
        return x

class Refinement_model(nn.Module):
    def __init__(self, stroke_num=512, stroke_dim=13, self_attn_depth=1,control_num=False,use_resnet=False):
        super(Refinement_model, self).__init__()
        self.encoder1 = path_predictor(0, stroke_num, stroke_dim, self_attn_depth, control_num=control_num,
                                    use_resnet=use_resnet, num_loss=True)
        self.encoder2 = path_predictor(1, 8, stroke_dim, self_attn_depth, control_num=control_num, use_resnet=use_resnet)

    def forward(self, x, canvas, step=0):
        if step == 0:
            return self.encoder1(x, canvas)
        else:
            return self.encoder2(x, canvas)

