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
#print(train_data.data.shape)
train_loader = data_utils.DataLoader(dataset=train_data, batch_size=100, shuffle=True)
test_loader = data_utils.DataLoader(dataset=test_data, batch_size=100, shuffle=True)

print(train_loader.dataset[0][0].shape)
print(train_loader)
for batch_idx, (data, target) in enumerate(train_loader):
    print("batch_idx:",batch_idx,",(data,target):",(data,target))

#这里有600个batch ，data里有100张图片，target里是这100张图片对应的标签