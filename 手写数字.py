import torch
import torch.nn as nn
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils

# 获取数据集
train_data = dataset.MNIST(root="D",
                           train=True,
                           transform=transforms.ToTensor(),
                           download=True
                           )
test_data = dataset.MNIST(root="D",
                          train=False,
                          transform=transforms.ToTensor(),
                          download=False
                          )
train_loader = data_utils.DataLoader(dataset=train_data, batch_size=100, shuffle=True)
test_loader = data_utils.DataLoader(dataset=test_data, batch_size=100, shuffle=True)

#print(train_loader.shape)

# 创建网络
class Net(torch.nn.Module):
    def __init__(self):#这是一个构造函数
        super().__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.bat2d = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.linear = nn.Linear(14 * 14 * 32, 70)
        self.tanh = nn.Tanh()
        self.linear1 = nn.Linear(70, 30)
        self.linear2 = nn.Linear(30, 10)

    def forward(self, x):#前向传播是需要自定义的，反向传播是自动的
        y = self.conv(x)
        y = self.bat2d(y)
        y = self.relu(y)
        y = self.pool(y)
        y = y.view(y.size()[0], -1)
        y = self.linear(y)
        y = self.tanh(y)
        y = self.linear1(y)
        y = self.tanh(y)
        y = self.linear2(y)
        return y


cnn = Net()
cnn = cnn.cuda()

# 损失函数
los = torch.nn.CrossEntropyLoss()

# 优化函数
optime = torch.optim.Adam(cnn.parameters(), lr=0.01)

# 训练模型
for epo in range(10):
    for i, (images, lab) in enumerate(train_loader):
        images = images.cuda()
        lab = lab.cuda()
        out = cnn(images)
        loss = los(out, lab)
        optime.zero_grad()
        loss.backward()
        optime.step()
        print("epo:{},i:{},loss:{}".format(epo + 1, i, loss))

# 测试模型
loss_test = 0
accuracy = 0
with torch.no_grad():
    for j, (images_test, lab_test) in enumerate(test_loader):
        images_test = images_test.cuda()
        lab_test = lab_test.cuda()
        out1 = cnn(images_test)
        loss_test += los(out1, lab_test)
        loss_test = loss_test / (len(test_data) // 100)
        _, p = out1.max(1)
        accuracy += (p == lab_test).sum().item()
        accuracy = accuracy / len(test_data)
        print("loss_test:{},accuracy:{}".format(loss_test, accuracy))