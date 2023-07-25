import torch
import torchvision
import torchvision.transforms as transforms
import imgshow

'''------------------------------------
下载数据集

'''
# 这里实际上做了一次张量的变换
# 下载数据集并对图⽚进⾏调整, 因为torchvision数据集的输出是PILImage格式, 数据域在[0,1]. 我们将其转换为标准数据域[-1, 1]的张量格式
#这里实际上做了一个归一化操作
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# shuffle 表示乱序
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# shuffle 表示乱序
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

'''随机抽取4个图片'''
dataiter = iter(trainloader)
print(type(dataiter))
print(dataiter.__class__)
print(dir(dataiter))
images, labels = dataiter.next()
imgshow.imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
