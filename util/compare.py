import os
import torch
from torchvision.utils import save_image
from torchvision import transforms
from lpips import LPIPS
from PIL import Image

model=LPIPS()
root='output'
loader=transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([512,512])
])
loss_mse=torch.nn.MSELoss()
mses=0
cnt=0
lpipss=0
dir='/home/huteng/SVG/AttnPainter/output/block=11/art'
#dir='/home/huteng/SVG/AttnPainter/util/Vector Magic'
for file in os.listdir(dir):
    print(file)
    file_path=os.path.join(dir,file)
    img0=loader(Image.open(file_path).convert('RGB'))
    img1=loader(Image.open(os.path.join('/home/huteng/SVG/AttnPainter/test-data/art/%s'%file)).convert('RGB'))
    mse=loss_mse(img0,img1)
    lpips_loss=model(img0,img1)
    mses+=mse
    lpipss+=lpips_loss
    cnt+=1
print(mses/cnt)
print(lpipss/cnt)
print(cnt)