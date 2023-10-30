import torch
import torch.nn as nn
from collections import OrderedDict
import timm 
from timm.models.vision_transformer import Block
from util.cross_attention import CrossAttentionBlock
from torchvision import transforms
from torchvision import models
class StrokeConvHeadBlock(nn.Module):
    def __init__(self, stroke_num, stroke_dim):
        super(StrokeConvHeadBlock, self).__init__()
        self.conv_layer = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)),
          ('relu1', nn.LeakyReLU()),
          ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
          ('relu2', nn.LeakyReLU()),
          ('conv3', nn.Conv2d(64, stroke_dim, kernel_size=1, stride=1, padding=1)),
          ('relu3', nn.LeakyReLU()),
          ('avgpool', nn.AdaptiveAvgPool2d((32, 32))),
        ]))
        self.linear_head = nn.Linear(1024, stroke_num)

    def forward(self, x):
        B, D, W, H = x.shape
        x = self.conv_layer(x).reshape(B, -1, 1024)
        return self.linear_head(x)


class StrokeConvHead(nn.Module):
    def __init__(self, stroke_num=[32, 64, 128, 256], stroke_dim=10):
        super(StrokeConvHead, self).__init__()
        self.layer_1 = StrokeConvHeadBlock(stroke_num[0], stroke_dim)
        self.layer_2 = StrokeConvHeadBlock(stroke_num[1], stroke_dim)
        self.layer_3 = StrokeConvHeadBlock(stroke_num[2], stroke_dim)
        self.layer_4 = StrokeConvHeadBlock(stroke_num[3], stroke_dim)

    def forward(self, x):
        return (self.layer_1(x[3]), self.layer_2(x[2]), self.layer_3(x[1]), self.layer_4(x[0]))
        

class StrokeConvPredictor(nn.Module):
    '''
    Convolution-based Stroke Predictor
    '''
    def __init__(self, stroke_num=[64, 128, 256, 512], stroke_dim=10):
        super(StrokeConvPredictor, self).__init__()
        self.feature_extractor = timm.create_model('tv_resnet50', pretrained=True)
        self.fpn_neck = FPN()
        self.stroke_head = StrokeConvHead(stroke_num=stroke_num, stroke_dim=stroke_dim)


    def extract_features(self, x):
        x = self.feature_extractor.conv1(x)
        x = self.feature_extractor.bn1(x)
        x = self.feature_extractor.act1(x)
        x = self.feature_extractor.maxpool(x)

        c1 = self.feature_extractor.layer1(x)
        c2 = self.feature_extractor.layer2(c1)
        c3 = self.feature_extractor.layer3(c2)
        c4 = self.feature_extractor.layer4(c3)
        return (c1, c2, c3, c4)

    def forward(self, x):
        feats = self.extract_features(x)
        feats = self.fpn_neck(feats)
        strokes = torch.cat(self.stroke_head(feats), dim=2).permute(0, 2, 1)
        return strokes

class StrokeAttentionHead(nn.Module):
    '''
    Cross-Attention-based Head
    '''
    def __init__(self, stroke_num=256, stroke_dim=13, encoder_embed_dim=768, self_attn_depth=4,control_num=False):
        super(StrokeAttentionHead, self).__init__()
        self.control_num=control_num
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
    
class StrokeAttentionEncoder(nn.Module):
    '''
    Cross-Attention-based Head
    '''
    def __init__(self, stroke_num=256, latent_dim=16, encoder_embed_dim=768, self_attn_depth=4):
        super(StrokeAttentionEncoder, self).__init__()

        self.stroke_tokens = nn.Parameter(torch.zeros(1, latent_dim, stroke_num))
        self.cross_attn_block = CrossAttentionBlock(x_dim=encoder_embed_dim, y_dim=stroke_num, num_heads=8)
        self.self_attn_blocks = nn.Sequential(*[
            Block(
                dim=stroke_num, num_heads=8, mlp_ratio=4., qkv_bias=True, drop=0.,
                attn_drop=0., norm_layer=nn.LayerNorm, act_layer=nn.GELU)
            for i in range(self_attn_depth)])
        self.norm = nn.LayerNorm(stroke_num)
        
    def forward(self, x):
        x = self.cross_attn_block(x, self.stroke_tokens.repeat(x.shape[0],1,1))
        x = self.self_attn_blocks(x)
        return self.norm(x).permute(0, 2, 1)
    
class StrokeAttentionDecoder(nn.Module):
    '''
    Cross-Attention-based Head
    '''
    def __init__(self, stroke_num=256, stroke_dim=13, latent_dim=16, self_attn_depth=2):
        super(StrokeAttentionEncoder, self).__init__()
        self.input_layer = nn.Linear(latent_dim, stroke_dim)
        self.self_attn_blocks = nn.Sequential(*[
            Block(
                dim=stroke_num, num_heads=8, mlp_ratio=4., qkv_bias=True, drop=0.,
                attn_drop=0., norm_layer=nn.LayerNorm, act_layer=nn.GELU)
            for i in range(self_attn_depth)])
        self.linear_head = nn.Linear(stroke_dim, stroke_dim)
        
    def forward(self, x):
        x = self.input_layer(x).permute(0, 2, 1)
        x = self.self_attn_blocks(x)
        x = self.linear_head(x.permute(0, 2, 1))
        return x
    
class StrokeAttentionPredictor(nn.Module):
    '''
    Cross-Attention-based Predictor
    '''
    def __init__(self, stroke_num=512, stroke_dim=13, self_attn_depth=1,control_num=False):
        super(StrokeAttentionPredictor, self).__init__()
        self.feature_extractor = timm.create_model('vit_small_patch16_224_dino', pretrained=False)
        self.feature_extractor.load_state_dict(torch.load('/home/huteng/.cache/torch/hub/checkpoints/dino_deitsmall16_pretrain.pth'))
        self.stroke_head = StrokeAttentionHead(stroke_num=stroke_num, stroke_dim=stroke_dim, encoder_embed_dim=self.feature_extractor.embed_dim, self_attn_depth=self_attn_depth,control_num=control_num)
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
        return torch.sigmoid(x)
        #return torch.sigmoid(x)*1.5-0.25

class StrokeAttentionPredictor_autoregression(nn.Module):
    '''
    Cross-Attention-based Predictor
    以target和canvas为输入，输出32笔
    '''
    def __init__(self, stroke_num=512, stroke_dim=13, self_attn_depth=1,control_num=False):
        super(StrokeAttentionPredictor_autoregression, self).__init__()
        #self.feature_extractor1 = timm.create_model('vit_small_patch16_224_dino', pretrained=False)
        # self.feature_extractor2 = timm.create_model('vit_small_patch16_224_dino', pretrained=False)
        # #self.stroke_head1 = StrokeAttentionHead(stroke_num=stroke_num, stroke_dim=stroke_dim, encoder_embed_dim=self.feature_extractor1.embed_dim, self_attn_depth=self_attn_depth,control_num=control_num)
        # self.stroke_head2 = StrokeAttentionHead(stroke_num=stroke_num, stroke_dim=stroke_dim,
        #                                        encoder_embed_dim=self.feature_extractor2.embed_dim,
        #                                        self_attn_depth=self_attn_depth,control_num=control_num)
        self.resize_224 = transforms.Resize((224, 224))
        self.map_in2 = nn.Sequential(
            nn.Conv2d(6, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 3, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )
        self.resnet=models.resnet50(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        print('num featrues',num_ftrs)
        self.resnet.fc = nn.Linear(num_ftrs, 16*stroke_dim)
    # def extract_features1(self, x):
    #     x = self.feature_extractor1.patch_embed(x)
    #     cls_token = self.feature_extractor1.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    #     x = torch.cat((cls_token, x), dim=1)
    #     x = self.feature_extractor1.pos_drop(x + self.feature_extractor1.pos_embed)
    #     x = self.feature_extractor1.blocks(x)
    #     x = self.feature_extractor1.norm(x)
    #     return x[:,1:]

    def extract_features2(self, x):
        x = self.feature_extractor2.patch_embed(x)
        cls_token = self.feature_extractor2.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.feature_extractor2.pos_drop(x + self.feature_extractor2.pos_embed)
        x = self.feature_extractor2.blocks(x)
        x = self.feature_extractor2.norm(x)
        return x[:,1:]

    def forward(self, x, canvas=None, step=1,num=32):
        if x.size(-1) != 224:
            x = self.resize_224(x)
            #canvas=self.resize_224(canvas)
        # if step==0:
        #     # x = torch.cat([x, canvas], dim=1)
        #     # x=self.map_in1(x)
        #     x = self.extract_features1(x)
        #     x = self.stroke_head1(x,num)
        # else:
            # x = torch.cat([x, canvas], dim=1)
        x = self.map_in2(x)
        x = self.resnet(x)
        x=x.view(x.size(0),16,-1)
        #x = torch.cat([x, canvas], dim=1)
        # x = self.map_in2(x)
        # x = self.extract_features2(x)
        # x = self.stroke_head2(x, num)
        return torch.sigmoid(x)

# class StrokeAttentionPredictor_autoregression(nn.Module):
#     '''
#     Cross-Attention-based Predictor
#     以target和canvas为输入，输出32笔
#     '''
#     def __init__(self, stroke_num=512, stroke_dim=13, self_attn_depth=1):
#         super(StrokeAttentionPredictor_autoregression, self).__init__()
#         self.feature_extractor = timm.create_model('vit_small_patch16_224_dino', pretrained=False)
#         self.stroke_head1 = StrokeAttentionHead(stroke_num=stroke_num, stroke_dim=stroke_dim, encoder_embed_dim=self.feature_extractor.embed_dim, self_attn_depth=self_attn_depth)
#         self.stroke_head2 = StrokeAttentionHead(stroke_num=stroke_num, stroke_dim=stroke_dim,
#                                                encoder_embed_dim=self.feature_extractor.embed_dim,
#                                                self_attn_depth=self_attn_depth)
#         self.resize_224 = transforms.Resize((224, 224))
#         self.map_in=nn.Sequential(
#             nn.Conv2d(6, 128, 3, 1, 1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.Conv2d(128, 3, 3, 1, 1),
#             nn.BatchNorm2d(3),
#             nn.ReLU(),
#         )
#     def extract_features(self, x):
#         x = self.feature_extractor.patch_embed(x)
#         cls_token = self.feature_extractor.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
#         x = torch.cat((cls_token, x), dim=1)
#         x = self.feature_extractor.pos_drop(x + self.feature_extractor.pos_embed)
#         x = self.feature_extractor.blocks(x)
#         x = self.feature_extractor.norm(x)
#         return x[:,1:]
#
#     def forward(self, x,canvas,step=0):
#         if x.size(-1) != 224:
#             x = self.resize_224(x)
#             canvas=self.resize_224(canvas)
#         x = torch.cat([x, canvas], dim=1)
#         x=self.map_in(x)
#         x = self.extract_features(x)
#         if step==0:
#             x = self.stroke_head1(x,32)
#         else:
#             x = self.stroke_head1(x, 32)
#         return torch.sigmoid(x)
class tmp_encoder(nn.Module):
    def __init__(self, stage,stroke_num=512, stroke_dim=13, self_attn_depth=1,control_num=False,use_resnet=False):
        super(tmp_encoder, self).__init__()

        self.stage=stage
        self.resize_224 = transforms.Resize((224, 224))
        if stage==0:
            self.feature_extractor = timm.create_model('vit_small_patch16_224_dino', pretrained=False)
            self.stroke_head = StrokeAttentionHead(stroke_num=stroke_num, stroke_dim=stroke_dim,
                                                   encoder_embed_dim=self.feature_extractor.embed_dim,
                                                   self_attn_depth=self_attn_depth, control_num=control_num)
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
                self.stroke_head = StrokeAttentionHead(stroke_num=stroke_num, stroke_dim=stroke_dim,
                                                       encoder_embed_dim=self.feature_extractor.embed_dim,
                                                       self_attn_depth=self_attn_depth, control_num=control_num)
            else:
                self.resnet=models.resnet50(pretrained=True)
                num_ftrs = self.resnet.fc.in_features
                print('num featrues',num_ftrs)
                self.resnet.fc = nn.Linear(num_ftrs, 16*stroke_dim)
    def extract_features(self, x):
        x = self.feature_extractor.patch_embed(x)
        cls_token = self.feature_extractor.cls_token.expand(x.shape[0], -1,
                                                             -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.feature_extractor.pos_drop(x + self.feature_extractor.pos_embed)
        x = self.feature_extractor.blocks(x)
        x = self.feature_extractor.norm(x)
        return x[:, 1:]

    def forward(self, x, canvas,  num=32):
        if x.size(-1) != 224:
            x = self.resize_224(x)
            canvas = self.resize_224(canvas)
        if self.stage==0:
            x = self.extract_features(x)
            x = self.stroke_head(x, num)
        else:
            if self.use_resnet:
                x = torch.cat([x, canvas], dim=1)
                x = self.map_in(x)
                x = self.resnet(x)
                x=x.view(x.size(0),16,-1)
            else:
                x = torch.cat([x, canvas], dim=1)
                x = self.map_in(x)
                x = self.extract_features(x)
                x = self.stroke_head(x, 16)
            #print('hhh')

        return torch.sigmoid(x)
class StrokeAttentionPredictor_autoregression2(nn.Module):
    '''
    Cross-Attention-based Predictor
    以target和canvas为输入，输出32笔
    两个encoder整合为一个
    '''
    def __init__(self, stroke_num=512, stroke_dim=13, self_attn_depth=1,control_num=False,use_resnet=False):
        super(StrokeAttentionPredictor_autoregression2, self).__init__()
        self.encoder1=tmp_encoder(0,stroke_num,stroke_dim,self_attn_depth,control_num=control_num,use_resnet=use_resnet)
        self.encoder2 = tmp_encoder(1, stroke_num, stroke_dim, self_attn_depth,control_num=control_num,use_resnet=use_resnet)

    def forward(self, x,canvas,step=0,num=32):
        if step==0:
            return self.encoder1(x,canvas,num)
        else:
            return self.encoder2(x, canvas, num)

class StrokeAttentionPredictorDen(nn.Module):
    '''
    Cross-Attention-based Predictor
    '''
    def __init__(self, stroke_num=512, stroke_dim=13, self_attn_depth=1, in_channels=4):
        super(StrokeAttentionPredictorDen, self).__init__()
        self.feature_extractor = timm.models.vision_transformer.vit_small_patch16_224(in_chans=in_channels)
        self.stroke_head = StrokeAttentionHead(stroke_num=stroke_num, stroke_dim=stroke_dim, encoder_embed_dim=self.feature_extractor.embed_dim, self_attn_depth=self_attn_depth)

    def extract_features(self, x):
        x = self.feature_extractor.patch_embed(x)
        cls_token = self.feature_extractor.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.feature_extractor.pos_drop(x + self.feature_extractor.pos_embed)
        x = self.feature_extractor.blocks(x)
        x = self.feature_extractor.norm(x)
        return x[:,1:]

    def forward(self, x):
        x = self.extract_features(x)
        x = self.stroke_head(x)
        return torch.sigmoid(x)

class Stroke_merge_head(nn.Module):
    '''
    Cross-Attention-based Head
    '''
    def __init__(self, stroke_num=128, stroke_dim=12*6+3, encoder_embed_dim=768, self_attn_depth=4,hidden_dim=512):
        super(Stroke_merge_head, self).__init__()

        self.blocks=nn.ModuleList()
        self.linear_in = nn.Linear(27, hidden_dim)
        self.linear_condition=nn.Linear(encoder_embed_dim,hidden_dim)
        self.stroke_tokens = nn.Parameter(torch.zeros(1, stroke_dim, stroke_num))
        self.linear_head=nn.Linear(hidden_dim,stroke_dim)
        for i in range(self_attn_depth):
            self.blocks.append(
            Block(dim=hidden_dim, num_heads=8, mlp_ratio=4., qkv_bias=True,
                attn_drop=0., norm_layer=nn.LayerNorm, act_layer=nn.GELU)
            )
            self.blocks.append(CrossAttentionBlock(y_dim=hidden_dim, x_dim=hidden_dim, num_heads=8))
        self.blocks.append(
            Block(dim=hidden_dim, num_heads=8, mlp_ratio=4., qkv_bias=True,
                  attn_drop=0., norm_layer=nn.LayerNorm, act_layer=nn.GELU)
        )
        self.cross_attn_block = CrossAttentionBlock(x_dim=196, y_dim=stroke_num, num_heads=4)
        self.self_attn_blocks = Block(
                dim=stroke_num, num_heads=4, mlp_ratio=4., qkv_bias=True,
                attn_drop=0., norm_layer=nn.LayerNorm, act_layer=nn.GELU)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(stroke_num)
    def forward(self, x,img_features):
        x=self.linear_in(x)
        img_features=self.linear_condition(img_features)
        for id,block in enumerate(self.blocks):
            if id%2==0:
                x=block(x)
            else:
                x=block(x,img_features)
        x=self.norm1(x)
        x = self.linear_head(x)
        return x[:,:128,:]
        return x



class Path_Transformer(nn.Module):
    '''
    Cross-Attention-based Predictor
    '''
    def __init__(self, stroke_num=512, stroke_dim=13, self_attn_depth=1):
        super(Path_Transformer, self).__init__()
        self.feature_extractor = timm.create_model('vit_small_patch16_224_dino', pretrained=False)
        self.feature_extractor.load_state_dict(torch.load('/home/huteng/.cache/torch/hub/checkpoints/dino_deitsmall16_pretrain.pth'))
        print('feature dim',self.feature_extractor.embed_dim)
        self.stroke_head = Stroke_merge_head(stroke_num=stroke_num, stroke_dim=stroke_dim, encoder_embed_dim=self.feature_extractor.embed_dim, self_attn_depth=self_attn_depth)
        self.resize_224 = transforms.Resize((224, 224))
    def extract_features(self, x):
        x = self.feature_extractor.patch_embed(x)
        cls_token = self.feature_extractor.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.feature_extractor.pos_drop(x + self.feature_extractor.pos_embed)
        x = self.feature_extractor.blocks(x)
        x = self.feature_extractor.norm(x)
        return x[:,1:]

    def forward(self, x,images):
        if images.size(-1) != 224:
            images = self.resize_224(images)
        image_features = self.extract_features(images)
        x = self.stroke_head(x,image_features)
        return torch.sigmoid(x)
if __name__=='__main__':
    x=torch.randn((32,512,27)).cuda()
    images=torch.randn((32,3,256,256)).cuda()
    model=Path_Transformer(stroke_num=128,stroke_dim=12*6+3).cuda()
    print(model(x,images))