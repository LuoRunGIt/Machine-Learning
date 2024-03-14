from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from torchinfo import summary  # 这个是额外安装的原本不在torch中
import pandas as pd
import os
from collections import Counter


# 创建卷积神经网络
class CNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        # 图像大小为201x81
        # 第一层卷积计算结果为197x77
        # 池化后为98x38 注意池化除不尽则下取整
        # 第二次卷积为94x34
        # 第二次池化为47x17
        # 第一层线性为47x17x64=51136
        self.conv2_drop = nn.Dropout2d()  # 随机失活
        # 数据调整为一维数据
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(51136, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # 最大池化，2表示池化层大小
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)


'''
# 指定设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('Using {} device'.format(device))
model = CNNet().to(device)

##训练
# cost function used to determine best parameters 定义损失函数
cost = torch.nn.CrossEntropyLoss()

learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

'''


# Create the validation/test function

def test(dataloader, model, device, cost):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for batch, (X, Y) in enumerate(dataloader):
            X, Y = X.to(device), Y.to(device)
            pred = model(X)

            test_loss += cost(pred, Y).item()
            correct += (pred.argmax(1) == Y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size

    print(f'\nTest Error:\nacc: {(100 * correct):>0.1f}%, avg loss: {test_loss:>8f}\n')


# Train the model
# 设置超参数
epoches = 15
batch_size = 50
learning_rate = 0.001


def main():
    # 将声谱图图像加载到数据加载器中进行训练
    data_path = './data/spectrograms'  # looking in subfolder train

    yes_no_dataset = datasets.ImageFolder(
        root=data_path,
        transform=transforms.Compose([transforms.Resize((201, 81)),
                                      transforms.ToTensor()
                                      ])
    )
    print(yes_no_dataset)

    # 基于每个音频类的文件夹自动创建图像类标签和索引。我们将使用class_to_idx来查看图像数据集的类映射。
    class_map = yes_no_dataset.class_to_idx

    print("\nClass category and index of the images: {}\n".format(class_map))

    # 划分训练集和测试集
    # split data to test and train
    # use 80% to train
    train_size = int(0.8 * len(yes_no_dataset))
    test_size = len(yes_no_dataset) - train_size
    yes_no_train_dataset, yes_no_test_dataset = torch.utils.data.random_split(yes_no_dataset, [train_size, test_size])

    print("Training size:", len(yes_no_train_dataset))
    print("Testing size:", len(yes_no_test_dataset))

    # 因为数据集是随机分割的，所以让我们对训练数据进行计数，以验证数据在“是”和“否”类别的图像之间的分布是否相当均匀。
    # labels in training set
    train_classes = [label for _, label in yes_no_train_dataset]
    Counter(train_classes)

    # 将数据加载到DataLoader中，并指定在训练迭代中如何划分和加载数据的批量大小。我们还将设置工作者的数量，以指定加载数据的子流程的数量。
    train_dataloader = torch.utils.data.DataLoader(
        yes_no_train_dataset,
        batch_size=15,
        num_workers=2,
        shuffle=True
    )
    # shuffle 表示打乱顺序

    test_dataloader = torch.utils.data.DataLoader(
        yes_no_test_dataset,
        batch_size=15,
        num_workers=2,
        shuffle=True
    )
    # print(type(train_dataloader))

    td = train_dataloader.dataset[0][0][0][0]
    print(td)

    # 指定设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cost = torch.nn.CrossEntropyLoss()
    print('Using {} device'.format(device))
    cnn = CNNet().to(device)
    # cnn 实例化
    cnn.train()
    size = len(train_dataloader.dataset)
    print(cnn)

    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    # 开始训练
    for epoch in range(epoches):
        print("进行第{}个epoch".format(epoch))
        for batch, (X, Y) in enumerate(train_dataloader):

            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            pred = cnn(X)
            loss = cost(pred, Y)
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')

    test(test_dataloader, cnn, device, cost)

    print('Done!')
    url = os.path.dirname(os.path.realpath(__file__)) + '/models/'
    if not os.path.exists(url):
        os.makedirs(url)
        # specify the model save name
    model_name = 'simple_model.pth'
    # save the model to file
    torch.save(cnn, url + model_name)


if __name__ == "__main__":
    main()
