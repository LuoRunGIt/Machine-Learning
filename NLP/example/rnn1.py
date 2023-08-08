# 人名分类器
# 导入包
# 构建RNN模型（LSTM,GRU）

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
# 57
print("n_letters:", n_letters)


# 函数作用是去掉语言中的一些重音标记
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


data_path = "./data/names/"


def readLines(filename):
    '''从指定文件中读取每一行加载到内容中形成列表'''

    # 打开指定⽂件并读取所有内容, 使⽤strip()去除两侧空⽩符, 然后以'\n'进⾏切分
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    # 对应每⼀个lines列表中的名字进⾏Ascii转换, 使其规范化.最后返回⼀个名字列表
    return [unicodeToAscii(line) for line in lines]


filename = data_path + "Chinese.txt"
lines = readLines(filename)
print(lines)

'''-------------------------------------------------------
构建人名和所属语言的列表的字典

如：{"English":["Lily","Susan"],"Chinese":["Zhang"]}

'''
category_lines = {}

# all_categories形如：["English","Chinese"]
all_categories = []

# 读取指定路径下所有txt文件，使用glob，path中可以使用的正则表达式
for filename in glob.glob(data_path + '*.txt'):
    # 获取每个文件的文件名
    # [0]表示文件名称，[1]为.txt
    category = os.path.splitext(os.path.basename(filename))[0]
    # print(category)
    all_categories.append(category)
    # 读取每个文件的内容，形成名字列表
    # 点评：能这么写的原因主要是所有名字都是按文件名进行了分类了
    lines = readLines(filename)
    # 按照对应的类别，将名字列表写入到category_lines字典中
    category_lines[category] = lines

# 查看类别总数
n_categories = len(all_categories)
print("n_categories:", n_categories)
# 随便查看其中的⼀些内容，[5:]从第五个开始 [:5]到第五个为止
print(category_lines['Italian'][:5])

'''人名转one-hot 张量 '''


def lineToTensor(line):
    """将⼈名转化为对应onehot张量表示, 参数line是输⼊的⼈名"""
    # ⾸先初始化⼀个0张量, 它的形状(len(line), 1, n_letters)， 总共多少字母(如ab 就是2个字母)，每个字母代表1，有多少字母就有多少n_letters(57)
    # 代表⼈名中的每个字⺟⽤⼀个1 x n_letters的张量表示.
    tensor = torch.zeros(len(line), 1, n_letters)
    # 遍历这个⼈名中的每个字符索引和字符
    for li, letter in enumerate(line):
        # 使⽤字符串⽅法find找到每个字符在all_letters中的索引
        # 它也是我们⽣成onehot张量中1的索引位置
        tensor[li][0][all_letters.find(letter)] = 1
    # 返回结果
    return tensor
#这里为啥要三个维度呢
line = "Bai"
line_tensor = lineToTensor(line)
print("line_tensot:", line_tensor)

'''-------------------------
模型构建
'''

