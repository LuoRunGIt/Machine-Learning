import torch
import torch.nn as nn

# 5是输入张量的维度，6是隐藏层张量的维度，即隐藏层神经元个数，1是隐含层数量
rnn = nn.RNN(5, 6, 1)
#1为序列长度，如一次一个字母，3为批次数量batch_size，5为input_size与上文5对应
input=torch.randn(1,3,5)
#1与num_layers: 隐含层的数量对应，3为batch_size，6为隐藏层维度
h0=torch.randn(1,3,6)
output,hn=rnn(input,h0)
print(input)
print(output)