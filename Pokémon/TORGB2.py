from PIL import Image
import numpy as np
L_path='E:\\BaiduNetdiskDownload\\baokemeng\\pokemon\\squirtle\\00000205.jpg'
L_image=Image.open(L_path)
out = L_image.convert("RGB")
img=np.array(out)
print(out.mode)
print(out.size)
print(img.shape)


#扫描一个文件夹下得所有图片，然后将非RGB格式的文件转换为RGB格式
