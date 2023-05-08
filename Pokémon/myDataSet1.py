'''
使用dataset类的继承来处理数据
'''
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

random_data = np.random.randn(10, 3)
# 初始化一个3x10的矩阵
print(random_data)


# 继承dataset类

class MyDataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


test_class = MyDataSet(random_data)

for i in range(len(test_class)):
    print(test_class[i])

train_size = int(len(test_class) * 0.7)
test_size = len(test_class) - train_size

# 转用这个类的好处在于我可以用torch中的算法进行测试集和数据集的划分，简直不要太舒服
train_dataset, test_dataset = torch.utils.data.random_split(test_class, [train_size, test_size])
print(len(train_dataset), type(train_dataset))
print(len(test_dataset), type(test_dataset))

for i in range(len(train_dataset)):
    print(train_dataset[i][i])

# 那么问题来了，如果我带了标签呢
# 怎么一一对应呢
