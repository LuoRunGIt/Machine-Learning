import torch
#这是一个张量练习的脚本

x=torch.arange(12)
print(x)
print(x.shape)
print(x.numel())#元素个数
X1=x.reshape(3,4)
X2=x.reshape(3,-1)
X3=x.reshape(-1,4)
print(X1,X2,X3)

x2=torch.zeros((2,3,4))
print(x2)

x3=torch.ones(2,2)
x4=torch.randn((3,3))
x5=torch.tensor([[1,2],[3,4]])
print(x3,x4,x5)
print(x.sum())