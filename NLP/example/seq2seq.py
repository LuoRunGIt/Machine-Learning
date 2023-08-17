'''
第⼀步: 导⼊必备的⼯具包.
第⼆步: 对持久化⽂件中数据进⾏处理, 以满⾜模型训练要求.
第三步: 构建基于GRU的编码器和解码器.
第四步: 构建模型训练函数, 并进⾏训练.
第五步: 构建模型评估函数, 并进⾏测试以及Attention效果分析

'''

'''步骤1 导入工具包'''
# 从io⼯具包导⼊open⽅法
from io import open
# ⽤于字符规范化
import unicodedata
# ⽤于正则表达式
import re
# ⽤于随机⽣成数据
import random
# ⽤于构建⽹络结构和函数的torch⼯具包
import torch
import torch.nn as nn
import torch.nn.functional as F
# torch中预定义的优化⽅法⼯具包
from torch import optim

# 设备选择, 我们可以选择在cuda或者cpu上运⾏你的代码
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''步骤2对持久化⽂件中数据进⾏处理, 以满⾜模型训练要求'''
# 起始标志
SOS_token = 0
# 结束标志
EOS_token = 1


class Lang:
    def __init__(self, name):
        """初始化函数中参数name代表传⼊某种语⾔的名字"""
        # 将name传⼊类中
        self.name = name
        # 初始化词汇对应⾃然数值的字典
        self.word2index = {}
        # 初始化⾃然数值对应词汇的字典, 其中0，1对应的SOS和EOS已经在⾥⾯了
        self.index2word = {0: "SOS", 1: "EOS"}
        # 初始化词汇对应的⾃然数索引，这⾥从2开始，因为0，1已经被开始和结束标志占⽤了
        self.n_words = 2

    def addSentence(self, sentence):
        """添加句⼦函数, 即将句⼦转化为对应的数值序列, 输⼊参数sentence是⼀条句⼦"""
        # 根据⼀般国家的语⾔特性(我们这⾥研究的语⾔都是以空格分个单词)
        # 对句⼦进⾏分割，得到对应的词汇列表
        for word in sentence.split(' '):
            # 然后调⽤addWord进⾏处理
            self.addWord(word)

    def addWord(self, word):
        """添加词汇函数, 即将词汇转化为对应的数值, 输⼊参数word是⼀个单词"""
        # ⾸先判断word是否已经在self.word2index字典的key中
        if word not in self.word2index:
            # 如果不在, 则将这个词加⼊其中, 并为它对应⼀个数值，即self.n_words
            self.word2index[word] = self.n_words
            # 同时也将它的反转形式加⼊到self.index2word中
            self.index2word[self.n_words] = word
            # self.n_words⼀旦被占⽤之后，逐次加1, 变成新的self.n_words
            self.n_words += 1


name = "eng"#类型为英文
sentence = "hello I am Jay"
engl = Lang(name)
engl.addSentence(sentence)
print("word2index:", engl.word2index)
print("index2word:", engl.index2word)
print("n_words:", engl.n_words)
