# 导⼊若⼲⼯具包
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# 定义⼀个简单的⽹络类
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义第⼀层卷积神经⽹络, 输⼊通道维度=1, 输出通道维度=6, 卷积核⼤⼩3*3
        self.conv1 = nn.Conv2d(1, 6, 3)
        # 定义第⼆层卷积神经⽹络, 输⼊通道维度=6, 输出通道维度=16, 卷积核⼤⼩3*3
        self.conv2 = nn.Conv2d(6, 16, 3)
        # 定义三层全连接⽹络
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 在(2, 2)的池化窗⼝下执⾏最⼤池化操作
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        # 计算size, 除了第0个维度上的batch_size
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

##输入参数
params = list(net.parameters())
print(len(params))
print(params[0].size())

# 模拟输入
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

# 损失函数
output = net(input)
# target这里表示的是损失值
target = torch.randn(10)
# 改变target的形状为⼆维张量, 为了和output匹配
target = target.view(1, -1)
criterion = nn.MSELoss()
loss = criterion(output, target)
print(loss)

# 反向传播
# Pytorch中执⾏梯度清零的代码
net.zero_grad()  # 防止梯度在grad中进行累加
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)
# Pytorch中执⾏反向传播的代码
loss.backward()
print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# 更新网络参数
# 通过optim创建优化器对象
optimizer = optim.SGD(net.parameters(), lr=0.01)
# 将优化器执⾏梯度清零的操作
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
# 对损失值执⾏反向传播的操作
loss.backward()
# 参数的更新通过⼀⾏标准代码来执⾏
optimizer.step()

# 那如果有多个神经网络呢？
