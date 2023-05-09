from PIL import Image
import torch
from torch.utils.data import Dataset


# 自定义数据集处理
class MyDataSet(Dataset):
    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):  # 返回数据集的个数
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])  # 返回路径下的PIL图像
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':  # 判断是否为 RGB 图像
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:  # transform 对 PIL 读取的图片处理
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels