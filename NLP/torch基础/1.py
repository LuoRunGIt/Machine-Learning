import torch

# 这样初始化的值是内存中随机的值
x = torch.empty(5, 3)
print(x)

# 有初始化
x1 = torch.rand(5, 3)
print(x1)

# 初始化0
x2 = torch.zeros(5, 3, dtype=torch.long)
print(x2)

# 直接通过数据创建张量
x = torch.tensor([2.5, 3.5])
print(x)

# 通过已有的⼀个张量创建相同尺⼨的新张量
# 利⽤news_methods⽅法得到⼀个张量
x = x.new_ones(5, 3, dtype=torch.double)
print(x)
# 利⽤randn_like⽅法得到相同张量尺⼨的⼀个新张量, 并且采⽤随机初始化来对其赋值
y = torch.randn_like(x, dtype=torch.float)
print(y)
print(x.size())