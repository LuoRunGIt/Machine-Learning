import torch
import torch.nn as nn

#5表示inputsize，6表示隐藏层张量h的特征维度大小（即隐藏层神经元个数），1表示隐含层数，默认激活函数是tanh
rnn=nn.RNN(5,6,1)
#1表示序列长度，3为batch_size 3个样本，5表示inputsize
input=torch.randn(1,3,5)
print(input)
#1表示隐含层数，3为batchsize，6为隐含层维度h
h0=torch.randn(1,3,6)
output, hn = rnn(input, h0)
print(output)