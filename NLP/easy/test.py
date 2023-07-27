import torch
import torchvision
import torchvision.transforms as transforms
#import cifar10

import imgshow
import torch.nn as nn
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=14, shuffle=False, num_workers=0)#调整这里的batch_size可以得到更多的结果
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
dataiter = iter(testloader)
images, labels = dataiter.next()
# 打印原始图⽚
imgshow.imshow(torchvision.utils.make_grid(images))
# 打印真实的标签
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(14)))

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
# ⾸先实例化模型的类对象
net = Net()
# 加载训练阶段保存好的模型的状态字典
PATH = './cifar_net.pth'
net.load_state_dict(torch.load(PATH))
# 利⽤模型对图⽚进⾏预测
outputs = net(images)
# 共有10个类别, 采⽤模型计算出的概率最⼤的作为预测的类别
_, predicted = torch.max(outputs, 1)
# 打印预测标签的结果
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(14)))


correct = 0
total = 0
with torch.no_grad():
 for data in testloader:
    images, labels = data
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 10000 test images: %d %%' % (
 100 * correct / total))