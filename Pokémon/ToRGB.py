import os
from PIL import Image
from tqdm import tqdm
import numpy as np
root_path='E:\\BaiduNetdiskDownload\\baokemeng\\pokemon\\mewtwo'
for item in tqdm():
    arr=item.strip().split('*')
    img_name=arr[0]
    image_path=os.path.join(root_path,img_name)
    img=Image.open(image_path)
    if(img.mode!='RGB'):
        img = img.convert("RGB")
        img=np.array(img)
        print(img_name)
        print(img.shape)

#把非RGB得图像转换为RGB图