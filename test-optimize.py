import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os
from models.attn_painter import AttnPainterSVG
data_root='/home/huteng/dataset/imagenet/val'
paths=[os.path.join(data_root,i) for i in os.listdir(data_root)]
paths=paths[-64:]
l=len(paths)
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    ])
img = Image.open(paths[1]).convert('RGB')
img = transform(img).unsqueeze(0).cuda()
width=256
model = AttnPainterSVG(stroke_num=128, path_num=4, width=width, control_num=True)
ckpt = torch.load('/home/huteng/SVG/AttnPainter/output_control_num/checkpoints2.pt')
model.load_state_dict(ckpt)
model.eval()
model=model.cuda()
strokes=torch.rand(1,16,27).cuda()
strokes.requires_grad=True
optimizer = torch.optim.AdamW([strokes], lr=0.005, betas=(0.9, 0.95))
canvas=torch.zeros_like(img).cuda()
gt_strokes=model.predict_path(img,num=64).detach()
print(gt_strokes.shape)
# #gt_strokes2=gt_strokes[:,:128,:]
# print(gt_strokes.shape)
canvas=model.rendering(gt_strokes[:,:32,:])[:,:3,:,:]
print(((canvas-img)**2).mean())
save_image(torch.cat([img,canvas[:,:3,:,:]],dim=0),'output/start.jpg')
canvas=model.rendering(gt_strokes)[:,:3,:,:]
print(((canvas-img)**2).mean())


for i in range(10000):
    optimizer.zero_grad()
    loss,loss0,loss_stroke,pred=model.loss_optimize(img,strokes,gt_strokes[:,32:,:],gt_strokes[:,:32,:])
    loss.backward()
    optimizer.step()
    if i %10==0:
        print(i,'loss:',loss.item(),'mse:',loss0.item(),'dtw',loss_stroke)
    if i%100==0:
        save_image(torch.cat([img,canvas[:,:3,:,:],pred[:,:3,:,:]],dim=0),'output/%d.jpg'%i)
# loss,loss0,canvas,pred=model.loss_optimize(img,strokes[:,:16,:])
# save_image(torch.cat([img,canvas[:,:3,:,:],pred[:,:3,:,:]],dim=0),'output/tmp1.jpg')
# loss,loss0,canvas,pred=model.loss_optimize(img,strokes[:,16:,:])
# save_image(torch.cat([img,canvas[:,:3,:,:],pred[:,:3,:,:]],dim=0),'output/tmp2.jpg')