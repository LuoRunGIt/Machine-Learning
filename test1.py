import torch
import torchvision
import cv2
from mxnet import autograd, nd
a1=torch.cuda.is_available()
print(a1)
a=torch.randn(3)
b=torch.randn(3,4)
print("a:",a)
print("b:",b)
x=nd.random.normal(scale=0.01, shape=(20,20))
print(x)