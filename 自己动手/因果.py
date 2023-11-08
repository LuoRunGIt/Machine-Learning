import matplotlib
import matplotlib.pyplot as plt

import torch
from torch import nn
from d2l import  torch as d2l

#使用正弦函数和一些噪声生成的序列数据
T=1000
time=torch.arange(1,T+1,dtype=torch.float32)
x=torch.sin(0.01*time)+torch.normal(0,0.2,(T,))
d2l.plot(time,[x],'time','x',xlim=[1,1000],figsize=(6,3))
plt.show()