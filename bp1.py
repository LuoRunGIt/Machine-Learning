# bp神经网络手敲
import numpy as np
import torch


# 定义激活函数
def Sigmoid(x):
    function = 1.0 / (1.0 + np.exp(-x))
    return function


# 对sigmod函数进行求导
def DS(x):
    f = Sigmoid(x)
    derivative = f * (1.0 - f)
    return derivative


# 三层神经网络，x为输入
def forward(x, w1, w2, b1, b2, xite):
    '''

    :param x: 输入
    :param w1: 输入层到隐藏层权值矩阵
    :param w2: 隐藏层到输出层权值矩阵
    :param b1: 偏置1
    :param b2: 偏置2
    :param xite: 学习率
    :return:
    '''

    # 这里有个问题，这里矩阵维度不同这是怎么相加的
    a1 = x#1x24
    #a2为1x25,w1.T 为24x25
    a2 = a1 @ w1.T + b1
    z2 = Sigmoid(a2)
    a3 = z2 @ w2.T + b2
    z3 = Sigmoid(a3)  # (1,4)
    return z2, z3, a2, a3

# 损失值
def loss(y_hat, y):
    # y
    # y.shape[0]指的是神经元的个数，为啥要乘以神经元的个数
    out = np.array((y - y_hat) ** 2) / 2 * y.shape[0]
    return out


# 求导
def DLoss(y_hat, y):
    out = np.array(y - y_hat)
    return out


# 反向传递函数
# x_hat表示真实值
# xtie 是学习率
def grad(out, out_hat, a2, a3, w1, w2, b1, b2, z2, x, xite):
    # 该函数实现参数更新,dloss中先后顺序不影响
    dw1_1 = DLoss(out, out_hat) * DS(a3)
    # 这里是矩阵乘啊
    dw1_2 = dw1_1 @ w2
    dw1_3 = dw1_2 * DS(a2)
    dw1 = dw1_3.T @ x

    dw2_1 = DLoss(out, out_hat) * DS(a3)
    dw2 = np.dot(dw2_1.T, z2)

    db1_1 = DLoss(out, out_hat) * DS(a3)
    db1_2 = db1_1 @ w2
    db1 = db1_2 * DS(a2)
    db2 = db1_1 * DS(a3)

    w2 = w2 - xite * dw2
    w1 = w1 - xite * dw1
    b2 = b2 - xite * db2
    b1 = b1 - xite * db1

    return w1, w2, b1, b2


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
# [2000,24]
n = 275  # 这个n用来干嘛
input1 = np.array([data[:, 1:25]])  # 分散数据集和标签，因为第一列为标签1，2，3，4
input1.resize(2000, 24)
# 2000行 24列
# 这里应该output为2000行1列还是1行2000列,估计是后者
output1 = np.array([data[:, 0]])

# 8000个固定间隔的数据
# 等于是个1行8000列的全0矩阵
output = np.linspace(0, 0, 8000)
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

data_num, _ = data.shape  # 得到样本数
index = np.arange(data_num)  # 生成下标
np.random.shuffle(index)  # 标签乱序
index = index

# 训练集
input_train = np.array(input1[index][0:1500, :])  # 左闭右开
output_train = np.array(output[index][0:1500, :])

# 测试集
input_test = np.array(input1[index][1500:2000, :])  # 左闭右开
output_test = np.array(output[index][1500:2000, :])
output_test1 = np.zeros((1, 500))  # 这个是干嘛的？

# 权重初始化

innum, midnum, outnum = 24, 25, 4

# https://blog.csdn.net/weixin_47156261/article/details/116611894
# 这里返回的是一个张量
w1 = np.array(torch.randn(midnum, innum))
w2 = np.array(torch.randn(outnum, midnum))
b1 = np.array(torch.zeros(1, midnum))
b2 = np.array(torch.zeros(1, outnum))

# 设置学习率
xite = 0.1
#print(input_train.shape)
print(input_train[1])
x1 = input_train[1].reshape(1, input_train.shape[1])
print(x1)
print(input_train[1].shape,x1.shape)
print(w1.T.shape)
e1=x1@w1.T
print(e1.shape)
# 神经网络循环过程
for j in range(10):
    #这里是 input_train是1500*24的
    #相当于循环1500次
    for i in range(input_train.shape[0]):
        #这步等于把一个数列24个数，转化为1行24列的矩阵
        x = input_train[i].reshape(1, input_train.shape[1])
        #
        z2, out, a2, a3 = forward(x, w1, w2, b1, b2, xite)
        # 反向传播
        w1, w2, b1, b2 = grad(out, output_train[1,:], a2, a3, w1, w2, b1, b2, z2, x, xite)

'------------------------------------------'
'--------------开始对模型准确率进行测试----------'
output_fore = np.zeros((500, 4))
out1 = []
list1 = []

for i in range(input_test.shape[0]):
    x = input_test[i].reshape(1, input_test.shape[1])
    z2, out, a2, a3 = forward(x, w1, w2, b1, b2,xite)
    # reshape(1,-1)表示转化为一行
    target = output_test[i].reshape(1, -1)

    output_fore[i, :] = out
    out1.append(target)
    list = np.argmax(out)
    list1.append(list)

list2 = np.zeros((500, 4))
for i in range(len(list1)):
    if list1[i] == 0:
        list2[i] = np.array([1, 0, 0, 0])
    elif list1[i] == 1:
        list2[i] = np.array([0, 1, 0, 0])
    elif list1[i] == 2:
        list2[i] = np.array([0, 0, 1, 0])
    elif list1[i] == 3:
        list2[i] = np.array([0, 0, 0, 1])

# 矩阵减法？
errorn = output_test - list2

output_list = np.zeros(2000)
output_list2 = np.zeros(500)

output_list = output1[0, index]
output_list2 = output_list[index][1500:2000]

for i in range(500):
    if output_list2[i] - list1[i] - 1 == 0:
        n += 1
print('准确率=', n / 500)
