import os

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from myDataSet import MyDataSet
from my_utils import read_split_data, plot_data_loader_image

# 数据集所在根目录,不需要划分trainSet+valSet，这里是完整数据集
#root = './data/flower'
root='E:\\BaiduNetdiskDownload\\baokemeng\\pokemon'
#E:\BaiduNetdiskDownload\baokemeng\pokemon\bulbasaur

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))  # 打印使用的设备

    # 划分训练集 + 验证集
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(root, val_rate=0.1,
                                                                                               flag=False)

    # 预处理
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),#随机裁剪，这里会导致一些图片很怪异，图片为224*224
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),#中心裁剪
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 数据处理
    train_data_set = MyDataSet(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transform["train"])

    val_data_set=MyDataSet(images_path=val_images_path,
                               images_class=val_images_label,
                               transform=data_transform["val"])

    batch_size = 8
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers'.format(nw))

    # 获取数据，测试的时候num_workers 设定为0
    train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True, num_workers=nw,
                              collate_fn=train_data_set.collate_fn)
    val_loader=DataLoader(val_data_set, batch_size=batch_size, shuffle=True, num_workers=nw,
                              collate_fn=train_data_set.collate_fn)
    # 可视化数据
    print("测试")
    plot_data_loader_image(train_loader)

    print("预测")
    plot_data_loader_image(val_loader)



if __name__ == '__main__':
    main()