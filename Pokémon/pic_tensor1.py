# 封装下函数
import torchvision.transforms as transforms
import cv2


def imgToTensor(imgpath):
    if imgpath == None:
        return None
    img = cv2.imread(imgpath)
    #这里先这么写
    img=cv2.resize(img, (1200, 1200))
    transf = transforms.ToTensor()
    img_tensor = transf(img)
    return img_tensor
