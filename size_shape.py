import torch

x = torch.arange(6)
X = x.reshape(2, 3)  # 转维度会在新的地址中
print(x)
print(X)
print(X.shape)  # 没有括号，没有括号，没有括号
print(X.size())  # 有括号
print(X.size)  # 直接指向内存地址
