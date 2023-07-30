# 导⼊torch和tensorboard的摘要写⼊⽅法
import torch
import json
import fileinput
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
import csv
#实例化写入对象
writer=SummaryWriter()

#随机初始化一个100*5的矩阵，将其视作已经得到的词嵌入矩阵
embedded=torch.randn(100,50)

# 导⼊事先准备好的100个中⽂词汇⽂件, 形成meta列表原始词汇

with open('vocab100.csv', 'r',encoding='utf-8') as read_obj:
    # Return a reader object which will
    # iterate over lines in the given csvfile
    csv_reader = csv.reader(read_obj)

    # convert string to list
    list_of_csv = list(csv_reader)

    print(list_of_csv)


print(len(list_of_csv))
writer.add_embedding(embedded,metadata= list_of_csv)
writer.close()