import matplotlib.pyplot as plt
import numpy as np
import torch

def f(x, alpha=0.1):
    return 1/(1+torch.exp(-(x-0.005)*1000))
# 定义 x 变量的范围 (-3，3) 数量 50
x=np.linspace(-1,1,1000)
x=torch.from_numpy(x)
print(f(torch.tensor([-0.001,0,0.001])))
y=f(x).numpy()
x=x.numpy()

# Figure 并指定大小
plt.figure(num=3,figsize=(8,5))
# 绘制 y=x^2 的图像，设置 color 为 red，线宽度是 1，线的样式是 --
plt.plot(x,y,color='red',linewidth=1.0,linestyle='--')
plt.savefig('tmp-draw.jpg')