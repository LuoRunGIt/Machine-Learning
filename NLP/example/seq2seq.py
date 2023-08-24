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


'''
name = "eng"  # 类型为英文
sentence = "hello I am Jay"
engl = Lang(name)
engl.addSentence(sentence)
print("word2index:", engl.word2index)
print("index2word:", engl.index2word)
print("n_words:", engl.n_words)
'''


# 将unicode转为Ascii, 我们可以认为是去掉⼀些语⾔中的重⾳标记：Ślusàrski
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    """字符串规范化函数, 参数s代表传⼊的字符串"""
    # 使字符变为⼩写并去除两侧空⽩符, z再使⽤unicodeToAscii去掉重⾳标记
    s = unicodeToAscii(s.lower().strip())
    # 在.!?前加⼀个空格
    s = re.sub(r"([.!?])", r" \1", s)
    # 使⽤正则表达式将字符串中不是⼤⼩写字⺟和正常标点的都替换成空格
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


'''
s = "Are you kidding me?"
nsr = normalizeString(s)
print(nsr)
'''

data_path = './data/eng-fra.txt'


def readLangs(lang1, lang2):
    """读取语⾔函数, 参数lang1是源语⾔的名字, 参数lang2是⽬标语⾔的名字
    返回对应的class Lang对象, 以及语⾔对列表"""
    # 从⽂件中读取语⾔对并以/n划分存到列表lines中
    lines = open(data_path, encoding='utf-8'). \
        read().strip().split('\n')
    # 对lines列表中的句⼦进⾏标准化处理，并以\t进⾏再次划分, 形成⼦列表, 也就是语⾔对
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    # 然后分别将语⾔名字传⼊Lang类中, 获得对应的语⾔对象, 返回结果
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)
    return input_lang, output_lang, pairs


lang1 = "eng"
lang2 = "fra"

input_lang, output_lang, pairs = readLangs(lang1, lang2)
print("input_lang:", input_lang)
print("output_lang:", output_lang)
# 此时所有数据并未转为index
print("word2index:", input_lang.word2index)
print("index2word:", input_lang.index2word)
# pairs中的前五个: [['go .', 'va !'], ['run !', 'cours !'], ['run !', 'courez !'], ['wow !', 'ca alors !'], ['fire !', 'au feu !']]
print("pairs中的前五个:", pairs[:5])

# 过滤出符合我们要求的语⾔对
# 设置组成句⼦中单词或标点的最多个数
MAX_LENGTH = 10
# 选择带有指定前缀的语⾔特征数据作为训练数据
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filterPair(p):
 """语⾔对过滤函数, 参数p代表输⼊的语⾔对, 如['she is afraid.', 'elle
malade.']"""
 # p[0]代表英语句⼦，对它进⾏划分，它的⻓度应⼩于最⼤⻓度MAX_LENGTH并且要以指定的前缀开头
 # p[1]代表法⽂句⼦, 对它进⾏划分，它的⻓度应⼩于最⼤⻓度MAX_LENGTH
 return len(p[0].split(' ')) < MAX_LENGTH and \
     p[0].startswith(eng_prefixes) and \
    len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
 """对多个语⾔对列表进⾏过滤, 参数pairs代表语⾔对组成的列表, 简称语⾔对列表"""
 # 函数中直接遍历列表中的每个语⾔对并调⽤filterPair即可
 return [pair for pair in pairs if filterPair(pair)]

fpairs = filterPairs(pairs)
print("过滤后的pairs前五个:", fpairs[:5])