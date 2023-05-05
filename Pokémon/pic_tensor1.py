# 封装下函数
import torchvision.transforms as transforms
import cv2
from skimage import io
#https://blog.csdn.net/qq_37924224/article/details/119181028
#png图片需要重新处理
def imgToTensor(imgpath):
    if imgpath == None:
        return None
    image = io.imread(imgpath)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    #这里先这么写
    img=cv2.resize(image, (1200, 1200))
    transf = transforms.ToTensor()
    img_tensor = transf(img)
    return img_tensor

