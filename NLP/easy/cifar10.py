import torch
import torchvision
import torchvision.transforms as transforms
import imgshow

'''------------------------------------
下载数据集

'''
# 这里实际上做了一次张量的变换
# 下载数据集并对图⽚进⾏调整, 因为torchvision数据集的输出是PILImage格式, 数据域在[0,1]. 我们将其转换为标准数据域[-1, 1]的张量格式
# 这里实际上做了一个归一化操作
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# shuffle 表示乱序
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# shuffle 表示乱序
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)
# 在windows下多线程往往存在问题
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

'''随机抽取4个图片'''
dataiter = iter(trainloader)
print(type(dataiter))
print(dataiter.__class__)
print(dir(dataiter))
images, labels = dataiter.next()
print("images===", images.shape)
print("labels===", labels)
imgshow.imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
print(labels[0])

'''卷积神经网络'''
import torch.nn as nn
import torch.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义2个卷积层
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # 池化层
        self.pool = nn.MaxPool2d(2, 2)

        # 全连接层

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        # 变换x的形状以适配全连接层
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
print(net)

'''定义损失函数'''
import torch.optim as optim

# 损失函数，交叉熵
criterion = nn.CrossEntropyLoss()
# 优化器，随机梯度下降,momentum动量
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

'''训练'''
# 整体数据集训练2次
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        # 梯度优化器归零
        optimizer.zero_grad()

        outputs = net(inputs)

        # 计算损失值
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 打印轮次和损失值
        running_loss += loss.item()
        if (i + 1) % 2000 == 0:
            print('[%d,%5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
    print("Finished Training")
# ⾸先设定模型的保存路径
PATH = './cifar_net.pth'
# 保存模型的状态字典
torch.save(net.state_dict(), PATH)

