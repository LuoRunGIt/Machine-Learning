import torch

x=torch.randn(4,4)
# 如果服务器上已经安装了GPU和CUDA
if torch.cuda.is_available():
 # 定义⼀个设备对象, 这⾥指定成CUDA, 即使⽤GPU
 device = torch.device("cuda")
 # 直接在GPU上创建⼀个Tensor
 y = torch.ones_like(x, device=device)
 # 将在CPU上⾯的x张量移动到GPU上⾯
 x = x.to(device)
 # x和y都在GPU上⾯, 才能⽀持加法运算
 z = x + y
 # 此处的张量z在GPU上⾯
 print(z)
 # 也可以将z转移到CPU上⾯, 并同时指定张量元素的数据类型
 print(z.to("cpu", torch.double))