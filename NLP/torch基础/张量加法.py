import torch
y = torch.rand(5, 3)
x=torch.ones(5,3)
print(y)

#加法
result=torch.empty(5,3)
torch.add(x,y,out=result)
print(result)

#print(torch.add(x, y))
#print(x + y)
y.add_(x)#这个相当于y被新的运算值取代了，同理下划线表示各种值置换
print(y)
print(x[:,:2])#打印5行2列