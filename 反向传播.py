import torch
'''
自动微分和反向传播的例子
'''
x=torch.arange(4.0)
print('x:',x)
#这里相当于将x.grad的内存锁住
x.requires_grad_(True)
print(x.grad,id(x.grad))
#2*x.T@x
y=2*torch.dot(x,x)
#结果为24
print('y:',y)
y.backward()#反向传播函数会计算梯度值，但不会更新参数值
print(x.grad,id(x.grad))

