import torch
x = torch.randn(4, 4)
# tensor.view()操作需要保证数据元素的总数量不变
y = x.view(16)
# -1代表⾃动匹配个数
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())



#如果张量中只有⼀个元素, 可以⽤.item()将值取出, 作为⼀个python numbere
x = torch.randn(1)
print(x)
print(x.item())

#张量转换为列表
y=torch.randn(2,2)
k=y.tolist()
print(k)
k1=y[0,0].tolist()
print(k1)