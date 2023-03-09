# 这个文件是对bp1中各种运算过程的拆解，应该会写好几个部分以体现出运算的各项细节
import numpy as np
import torch


def read_data(dir_str):
    data_temp = []
    with open(dir_str) as fdata:
        while True:
            line = fdata.readline()
            if not line:
                break
            data_temp.append([float(i) for i in line.split()])
    return np.array(data_temp)


data = read_data("语音数据txt.txt")
# print(data)
# print(data.shape)
# data 为2000行 25列的数据集合，其中
# print(data[:,0]) #表示第1列
# print(data[0,:] #表示第1行

input1 = np.array([data[:, 1:25]])  # 分散数据集和标签，因为第一列为标签1，2，3，4
input1.resize(2000, 24)

# (1,2000)这里相当于是1行2000列
output1 = np.array([data[:, 0]])
# print(output1.shape)
# print(output1)
# 8000个固定间隔的数据
# 等于是个1行8000列的全0矩阵
output = np.linspace(0, 0, 8000)
# print(output)
# print(output.shape)
# 8000行1列
output.resize(8000, 1)
# print(output)
# 2000行4列
output.resize(2000, 4)

for i in range(2000):
    if np.array(output1[:, i]) == 1:
        output[i:, ] = np.array([1, 0, 0, 0])
    elif np.array(output1[:, i]) == 2:
        output[i:, ] = np.array([0, 1, 0, 0])
    elif np.array(output1[:, i]) == 3:
        output[i:, ] = np.array([0, 0, 1, 0])
    elif np.array(output1[:, i]) == 4:
        output[i:, ] = np.array([0, 0, 0, 1])
# (2000 ,4 )
# print(output)
# print(output.shape)

data_num, _ = data.shape  # 得到样本数
# print(data.shape)# 2000.25
index = np.arange(data_num)  # 生成下标
# print(index)# 这个效果0，1，2，3，.....1999
np.random.shuffle(index)  # 标签乱序
index = index
# 打乱这些标签的顺序
# print(index)

# 训练集
# 根据index中的0-1499个赋值给
# 训练集24个特征
input_train = np.array(input1[index][0:1500, :])  # 左闭右开
# 训练集的输出，通过index是一一对应的
output_train = np.array(output[index][0:1500, :])

# print(input_train)
# 1500，24
# print(input_train.shape)
# 结果集合
# print(output_train)
# 1500，4
# print(output_train.shape)


# 权重初始化

innum, midnum, outnum = 24, 25, 4

# https://blog.csdn.net/weixin_47156261/article/details/116611894
# 这里返回的是一个张量
w1 = np.array(torch.randn(midnum, innum))#25x24
w2 = np.array(torch.randn(outnum, midnum))#4x25
b1 = np.array(torch.zeros(1, midnum))#1,25
b2 = np.array(torch.zeros(1, outnum))#1,4
# torch.randn:用来生成随机数字的tensor，
# 这些随机数字满足标准正态分布（0~1）。
print(w1)
# 25行24列
print(w1.shape)
