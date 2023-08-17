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


