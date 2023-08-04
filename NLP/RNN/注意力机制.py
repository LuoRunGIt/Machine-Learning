import torch
import torch.nn as nn
import torch.nn.functional as F

'''
第⼀步: 根据注意⼒计算规则, 对Q，K，V进⾏相应的计算.
第⼆步: 根据第⼀步采⽤的计算⽅法, 如果是拼接⽅法，则需要将Q与第⼆步的计算结果再进
⾏拼接, 如果是转置点积, ⼀般是⾃注意⼒, Q与V相同, 则不需要进⾏与Q的拼接.
第三步: 最后为了使整个attention机制按照指定尺⼨输出, 使⽤线性层作⽤在第⼆步的结果上
做⼀个线性变换, 得到最终对Q的注意⼒表示.
'''


class Attn(nn.Module):
    def __init__(self, query_size, key_size, value_size1, value_size2, output_size):
        """初始化函数中的参数有5个,
        query_size代表query的最后⼀维⼤⼩
         key_size代表key的最后⼀维⼤⼩,
         value_size1代表value的导数第⼆维⼤⼩,
         value = (1, value_size1, value_size2)
         value_size2代表value的倒数第⼀维⼤⼩,
         output_size输出的最后⼀维⼤⼩"""
        super(Attn, self).__init__()#这句初始化非常重要
        self.query_size = query_size
        self.key_size = key_size
        self.value_size1 = value_size1
        self.value_size2 = value_size2
        self.output_size = output_size

        # 初始化注意⼒机制实现第⼀步中需要的线性层
        self.attn = nn.Linear(self.query_size + self.key_size, value_size1)
        # 初始化注意⼒机制实现第三步中需要的线性层.
        self.attn_combine = nn.Linear(self.query_size + value_size2, output_size)

    def forward(self, Q, K, V):

        """forward函数的输⼊参数有三个, 分别是Q, K, V, 根据模型训练常识, 输⼊给Attion机制的
         张量⼀般情况都是三维张量, 因此这⾥也假设Q, K, V都是三维张量"""
        # 第⼀步, 按照计算规则进⾏计算,
        # 我们采⽤常⻅的第⼀种计算规则
        # 将Q，K进⾏纵轴拼接, 做⼀次线性变化, 最后使⽤softmax处理获得结果

        attn_weights = F.softmax(self.attn(torch.cat((Q[0], K[0]), 1)), dim=1)
        print('catshape===',torch.cat((Q[0],K[0]),1).shape)
        # 然后进⾏第⼀步的后半部分, 将得到的权重矩阵与V做矩阵乘法计算,
        # 当⼆者都是三维张量且第⼀维代表为batch条数时, 则做bmm运算
        # unsqueeze函数就是在指定的位置插入一个维度
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), V)
        # 之后进⾏第⼆步, 通过取[0]是⽤来降维, 根据第⼀步采⽤的计算⽅法,
        # 需要将Q与第⼀步的计算结果再进⾏拼接
        output = torch.cat((Q[0], attn_applied[0]), 1)
        # 最后是第三步, 使⽤线性层作⽤在第三步的结果上做⼀个线性变换并扩展维度，得到输出
        # 因为要保证输出也是三维张量, 因此使⽤unsqueeze(0)扩展维度
        output = self.attn_combine(output).unsqueeze(0)
        return output, attn_weights
        #attn_weights为注意力矩阵

query_size = 32
key_size = 32
value_size1 = 32
value_size2 = 64
output_size = 64
attn = Attn(query_size, key_size, value_size1, value_size2, output_size)
Q = torch.randn(1, 1, 32)
K = torch.randn(1, 1, 32)
V = torch.randn(1, 32, 64)
out = attn(Q, K, V)
print(out[0])
print(out[1])
