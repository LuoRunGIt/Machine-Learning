# 这里还是依靠手写数字识别去了解dataset的实际情况
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt  # %matplotlib inline 可以让Jupyter Notebook直接输出图像
import pylab  # 但我写上%matplotlib inline就报错 所以我就用了pylab.show()函数显示图像

# 接着定义一些训练用的超参数：
image_size = 28  # 图像的总尺寸为 28x28
num_classes = 10  # 标签的种类数
num_epochs = 20  # 训练的总猜环周期
batch_size = 64  # 一个批次的大小，64张图片

'''______________________________开始获取数据的过程______________________________'''
# 加载MNIST数据 MNIST数据属于 torchvision 包自带的数据,可以直接接调用
# 当用户想调用自己的图俱数据时，可以用torchvision.datasets.ImageFolder或torch.utils.data. TensorDataset来加载
train_dataset = dsets.MNIST(root='../D/MNIST/raw',  # 文件存放路径
                            train=True,  # 提取训练集
                            # 将图像转化为 Tensor，在加载數据时，就可以对图像做预处理
                            transform=transforms.ToTensor(),
                            download=True)  # 当找不到文件的时候，自动下載

# 加载测试数据集
test_dataset = dsets.MNIST(root='../D/MNIST/raw',
                           train=False,
                           transform=transforms.ToTensor())

train_data_size = len(train_dataset)
test_data_size = len(test_dataset)
print("训练集", "测试集", train_data_size, test_data_size)  # 60000,10000
print("训练集类型", type(train_dataset))  # train_dataset是一个类，应该是继承了dataset
# 训练数据集的加载器，自动将数据切分成批，顺序随机打乱
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
'''                                         
将测试数据分成两部分，一部分作为校验数据，一部分作为测试数据。
校验数据用于检测模型是否过拟合并调整参数，测试数据检验整个模型的工作
'''
# 首先，定义下标数组 indices，它相当于对所有test_dataset 中数据的编码
# 然后，定义下标 indices_val 表示校验集数据的下标，indices_test 表示测试集的下标
indices = range(len(test_dataset))  # 这里是int类型
print("indices", indices, type(indices))  # <class 'range'>
indices_val = indices[: 5000]
indices_test = indices[5000:]
# 根据下标构造两个数据集的SubsetRandomSampler 来样器，它会对下标进行来样
sampler_val = torch.utils.data.sampler.SubsetRandomSampler(indices_val)
sampler_test = torch.utils.data.sampler.SubsetRandomSampler(indices_test)
# 根据两个采样器定义加载器
# 注意将sampler_val 和sampler_test 分别賦值给了 validation_loader 和 test_loader
validation_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                sampler=sampler_val)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          sampler=sampler_test)
# 随便从数据集中读入一张图片，并绘制出来
idx = 70  # random.randint(1, 100)
print("dataset的类型",type(train_dataset))
# dataset支特下标索引，其中提取出来的元素为 features、target 格式，即属性和标签。[0]表示索引 features
muteimg = train_dataset[idx][0].numpy()
print(train_dataset)
# 一般的图像包含RGB 这了个通道，而 MNIST 数据集的因像都是交度的，只有一个通道
# 因此，我们忽略通道，把图像看作一个灰度矩阵
# 用 imshow 画图，会将交度矩阵自动展现为彩色，不同灰度对应不同的颜色：从黄到紫
# 从这里可以发现train_dataset [0]是像素点，[1]是分类结果
# 这个dataset为tuple数据结构
print("图", "标签", train_dataset[70][0], train_dataset[70][1])
print(train_dataset[70][0].shape)  # 图片维度是3 因为是28x28,注意shape没有括号
plt.imshow(muteimg[0, ...], cmap='gray_r')  # 不加cmap='gray_r'的话颜色偏蓝色
plt.title("{}: {}".format('image sample', train_dataset[idx][1]))  # 显示获取到的图片的标签
pylab.show()
