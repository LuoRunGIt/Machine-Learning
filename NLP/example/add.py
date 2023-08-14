import torch

a=torch.randn(4)
print(a)
print('-----------------')
b=torch.randn(4,1)
print(b)
print('-----------------')

c=torch.add(a,b)#求出是4x4的相当于a和b相加

print(c)
print('-----------------')
d=torch.add(a,b,alpha=10)#把bx10再执行加法
print(d)