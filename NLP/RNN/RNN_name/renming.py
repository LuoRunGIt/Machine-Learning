#RNN构建的人名分类器
'''
第⼀步: 导⼊必备的⼯具包.
第⼆步: 对data⽂件中的数据进⾏处理，满⾜训练要求.
第三步: 构建RNN模型(包括传统RNN, LSTM以及GRU).
第四步: 构建训练函数并进⾏训练.
第五步: 构建评估函数并进⾏预测.
'''

from io import open
import glob
import os
import string
import unicodedata
import random
import time
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# 获取所有常⽤字符包括字⺟和常⽤标点
all_letters = string.ascii_letters + ".,;'"
# 获取常⽤字符数量
n_letters = len(all_letters)
# 56,26个英文字母的大小写和.,;’
#print(all_letters)
#print("n_letters:", n_letters)

# 函数作用是去掉语言中的一些重音标记
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )