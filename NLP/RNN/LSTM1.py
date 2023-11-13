import torch
import torch.nn as nn

'''
第一个参数：input_size(输入张量x的维度)
第二个参数 hidden_size 隐藏层维度，隐藏层神经元个数
第三个参数 num_layer 隐藏层层数
'''

rnn=nn.LSTM(5,6,2)#bidirectional=True 设置双向
#rnn=nn.LSTM(5,6,2,bidirectional=True)#bidirectional=True 设置双向
'''
第一个参数 squence_length 输入序列的长度
第二个参数 batch_size 批次的样本数量
第三个参数 input_size 输入张量x的维度
'''

input=torch.randn(1,3,5)

'''
第一个参数 num_layer*num_directions (隐藏层层数*方向数) 
第二个参数 batch_size
第三个参数 num_layer
'''

h0=torch.randn(2,3,6)
c0=torch.randn(2,3,6)

output,(hn,cn)=rnn(input,(h0,c0))
print('lstmoutput===',output.shape)