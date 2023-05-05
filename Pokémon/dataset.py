# https://blog.csdn.net/jameschen9051/article/details/119515204
# 一个利用cnn实现宝可梦分类模型的实验
# 使用pytouch进行实现

# 步骤1.数据预处理
# 图片位置E:\BaiduNetdiskDownload\baokemeng\pokemon
# 使用张量
import matplotlib.pyplot as plt
import torch
import csv

# 由于文件位置和分类结果都在csv中所以需要先把图片位置读出来
# 以手写字体输入实验为例，输入为（batch_size，通道数，图像h,图像w）
# 首先我们需要将图片转换为tensor 并且和标签一一对应

import csv
csv_reader = csv.reader(open("E:\\BaiduNetdiskDownload\\baokemeng\\pokemon\\images.csv"))
for row in csv_reader:
	print(row)
#<class '_csv.reader'> 这里read的结果是一个结构体或者说类
print(type(csv_reader))
#['pokemon\\charmander\\00000016.png', '1']
#下一步要将这个链表结构数据转换为张量