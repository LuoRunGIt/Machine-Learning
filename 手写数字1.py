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
train_dataset = dsets.MNIST(root='./D/MNIST/raw',  # 文件存放路径
                            train=True,  # 提取训练集
                            # 将图像转化为 Tensor，在加载數据时，就可以对图像做预处理
                            transform=transforms.ToTensor(),
                            download=True)  # 当找不到文件的时候，自动下載

# 加载测试数据集
test_dataset = dsets.MNIST(root='./D/MNIST/raw',
                           train=False,
                           transform=transforms.ToTensor())

train_data_size = len(train_dataset)
test_data_size = len(test_dataset)
print(train_data_size, test_data_size)  # 60000,10000

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
indices = range(len(test_dataset))
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

# dataset支特下标索引，其中提取出来的元素为 features、target 格式，即属性和标签。[0]表示索引 features
muteimg = train_dataset[idx][0].numpy()
print(train_dataset)
# 一般的图像包含RGB 这了个通道，而 MNIST 数据集的因像都是交度的，只有一个通道
# 因此，我们忽略通道，把图像看作一个灰度矩阵
# 用 imshow 画图，会将交度矩阵自动展现为彩色，不同灰度对应不同的颜色：从黄到紫
# 从这里可以发现train_dataset [0]是像素点，[1]是分类结果
# 这个dataset为tuple数据结构
print(train_dataset[70][0], train_dataset[70][1])
print(train_dataset[70][0].shape)  # 图片维度是3 因为是28x28,注意shape没有括号
plt.imshow(muteimg[0, ...], cmap='gray_r')  # 不加cmap='gray_r'的话颜色偏蓝色
plt.title("{}: {}".format('image sample', train_dataset[idx][1]))  # 显示获取到的图片的标签
pylab.show()

'''
-------------------分隔符，上面是读取数据，下面是训练模型--------------
'''
depth = [4, 8]  # 模型深度


class ConvNet(nn.Module):
    def __init__(self):
        ##该函数在创建一个ConvNet 对象时即调用语句 net=ConvNet()时就会被调用，即初始化
        # 首先调用父类的相应的构造函数
        super(ConvNet, self).__init__()
        # 构造各个神经模块
        # 定义一个卷积层，输入通道为1（灰度图），输出为4 kernel为5,默认步长为1 填充为2
        self.conv1 = nn.Conv2d(1, 4, 5, padding=2)
        self.poo1 = nn.MaxPool2d(2, 2)  # 2d表示2维度,定义一个2x2的池化层
        self.conv2 = nn.Conv2d(depth[0], depth[1], 5, padding=2)
        # 一个线性连接层，输入尺寸为最后一层立方体的线性平铺，输出层512个节点
        # //表示整数除法
        self.fc1 = nn.Linear(image_size // 4 * image_size // 4 * depth[1], 512)
        # 最后一层输入512，输出10（0-9）
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):  # x应该是输入
        # x的size为（batch_size,imagel_channel，image_width,image_height）
        x = self.conv1(x)
        # F是 torchvision.transforms
        x = F.relu(x)  # 激活函数用Relu，防止过拟合
        # x的尺寸：（batch_size,num_filters,image_width/2,image_height/2）
        x = self.poo1(x)  # x进一步变小
        # x的尺寸：(batch_size, depth[0], image_width/ 2， image_height/2)

        x = self.conv2(x)  # 第三层又是卷积，窗口为5，输入输出通道分列为depth[o]=4,depth[1]=8
        x = self.poo1(x)  # 第四层池化，将图片缩小到原来的 1/4
        # x的尺寸：(batch_size, depth[1], image_width/ 4, image_height/4)
        # 将立体的特征图 tensor 压成一个一维的向量
        # view 函数可以将一个tensor 按指定的方式重新排布
        # 下面这个命令就是要让x按照batch_size * (image_ size//4)^2*depth[1]的方式来排布向量
        x = x.view(-1, image_size // 4 * image_size // 4 * depth[1])
        # x的尺寸：(batch_ size， depth[1]*image width/4*image height/4)
        x = F.relu(self.fc1(x))  # 第五层为全连接，ReLU激活函数
        # x的尺寸：(batch_size, 512)
        # 以默认0.5的概率对这一层进行dropout操作，防止过拟合
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)  # 全连接
        # X的尺寸：(batch_size, num_classes)
        # 输出层为 log_Softmax，即概率对数值 log(p(×))。采用log_softmax可以使后面的交叉熵计算更快
        x = F.log_softmax(x, dim=1)
        return x

    def retrieve_features(self, x):
        # 该函数用于提取卷积神经网络的特征图，返回feature_map1,feature_map2为前两层卷积层的特征图
        feature_map1 = F.relu(self.conv1(x))  # 完成第一层卷积
        x = self.pool(feature_map1)  # 完成第一层池化
        # 第二层卷积，两层特征图都存储到了 feature_map1,feature map2 中
        feature_map2 = F.relu(self.conv2(x))
        return (feature_map1, feature_map2)


'''-----网络构造完成------'''

'''开始训练'''
net = ConvNet()  # 初始化网络
criterion = nn.CrossEntropyLoss()  # 损失函数 交叉熵
## 定义优化器，普通的随机梯度下降算法
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
record = []  # 记录准确率等数值的容器
weights = []  # 每若干步就记录一次卷积核

#准确率
def rightness(output, target):
    preds = output.data.max(dim=1, keepdim=True)[1]
    return preds.eq(target.data.view_as(preds)).cpu().sum(), len(target)

#训练循环
for epoch in range(num_epochs):
    train_rights = []  # 记录训练数据集准确率的容器
    '''
    下面的enumerate起到构道一个枚举器的作用。在对train_loader做循环选代时，enumerate会自动输出一个数宇指示循环了几次，
    并记录在batch_idx中，它就等于0，1，2，... 
    train_loader 每选代一次，就会输出一对数据data和target，分别对应一个批中的手写数宇图及对应的标签。
    '''
    for batch_idx, (data, target) in enumerate(train_loader):  # 针对容器中的每一个批进行循环
        # 将 Tensor 转化为 Variable, data 为一批图像，target 为一批标签
        data, target = Variable(data), Variable(target)
        #64表示图片数量
        #<class 'torch.Tensor'> torch.Size([64, 1, 28, 28]) <class 'torch.Tensor'> torch.Size([64])
        #print(type(data),data.shape,type(target),target.shape)
        # 给网络模型做标记，标志着模型在训练集上训练
        # 这种区分主要是为了打开关闭net的training标志，从而决定是否运行dropout
        net.train()

        output = net(data)  # 神经网络完成一次前馈的计算过程，得到预测输出output
        loss = criterion(output, target)  # 将output与标签target比较，计算误差
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 一步随机梯度下降算法
        right = rightness(output, target)  # 计算准确率所需数值，返回数值为（正确样例数，总样本数）
        train_rights.append(right)  # 将计算结果装到列表容器train_rights中

        if batch_idx % 100 == 0:  # 每间隔100个batch 执行一次打印操作
            net.eval()  # 给网络楧型做标记，标志着模型在训练集上训练
            val_rights = []  # 记录校验数据集准确率的容器

            # 开始在校验集上做循环，计算校验集上的准确度
            for (data, target) in validation_loader:
                data, target = Variable(data), Variable(target)
                # 完成一次前馈计算过程，得到目前训练得到的模型net在校验集上的表现
                output = net(data)
                # 计算准确率所需数值，返回正确的数值为（正确样例数，总样本数）
                right = rightness(output, target)
                val_rights.append(right)
            # 分别计算目前已经计算过的测试集以及全部校验集上模型的表现：分类准确率
            # train_r为一个二元组，分别记录经历过的所有训练集中分类正确的数量和该集合中总的样本数
            # train_r[0]/train_r[1]是训练集的分类准殖度，val_[0]/val_r[1]是校验集的分类准确度
            train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
            # val_r为一个二元组，分别记录校验集中分类正确的数量和该集合中总的样本数
            val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))

            # 打印准确率等数值，其中正确率为本训练周期epoch 开始后到目前批的正确率的平均值
            print(val_r)
            print('训练周期: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t训练正确率: {:.2f}%\t校验正确率: {:.2f}%'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                loss.data,
                       100. * train_r[0].numpy() / train_r[1],
                       100. * val_r[0].numpy() / val_r[1]))

            # 将准确率和权重等数值加载到容器中，方便后续处理
            record.append((100 - 100. * train_r[0] / train_r[1], 100 - 100. * val_r[0] / val_r[1]))

            # weights 记录了训练周期中所有卷积核的演化过程，net.conv1.weight 提取出了第一层卷积核的权重
            # Clone 是将weight.data 中的数据做一个备份放到列表中
            # 否则当 weight.data 变化时，列表中的每一项数值也会联动
            # 这里使用clone这个函数很重要
            weights.append([net.conv1.weight.data.clone(), net.conv1.bias.data.clone(),
                            net.conv2.weight.data.clone(), net.conv2.bias.data.clone()])
#注意池化是poo1不是pool
#训练轮数为19

#print(type(train_dataset),train_dataset.shape)