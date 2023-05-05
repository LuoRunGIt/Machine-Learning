'''
用cv2读取图片保存为tansor
'''

import cv2
import torchvision.transforms as transforms

img=cv2.imread('E:/BaiduNetdiskDownload/baokemeng/pokemon/mewtwo/00000239.jpg')
print("cv2 读取后的图片维度",img.shape)
#(1200, 1600, 3) 对应通道为H,W,C
cv2.imshow("",img)
cv2.waitKey(0)

#初始化一个转换器
transf=transforms.ToTensor()
img_tensor=transf(img)
#torch.Size([3, 1200, 1600]) 对应H,W,C
print(img_tensor.shape)

#这里有一个问题，就是ToTensor 会把图像正则化，即压缩到【0，1】的范围内，这个目前我还不知道是否会影响计算

print(img_tensor)

#这里是吧图片归一化，相当于把图片范围划分到【-1，1】之间
img = transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])(img_tensor)
print(img)

#E:/BaiduNetdiskDownload/baokemeng/pokemon/squirtle/00000154.jpg

