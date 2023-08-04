'''
nn.GRU
input_size: 输⼊张量x中特征维度的⼤⼩.
hidden_size: 隐层张量h中特征维度的⼤⼩.
num_layers: 隐含层的数量.
bidirectional: 是否选择使⽤双向LSTM, 如果为True, 则使⽤; 默认不使⽤.

输出
input: 输⼊张量x.
h0: 初始化的隐层张量h
'''
import torch
import torch.nn as nn

rnn = nn.GRU(5, 6, 2)
input = torch.randn(1, 3, 5)
h0 = torch.randn(2, 3, 6)
output, hn = rnn(input, h0)
