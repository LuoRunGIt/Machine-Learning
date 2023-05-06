'''
这里将 文件路径对应的图片读取出来
'''
import csv

import torch

import pic_tensor1
import tansor_case

#csv_reader = csv.reader(open("E:\\BaiduNetdiskDownload\\baokemeng\\pokemon\\images.csv"))
csv_reader = csv.reader(open("E:\\BaiduNetdiskDownload\\baokemeng\\pokemon\\images1.csv"))

'''
csv.reader：以列表的形式返回读取的数据。
csv.writer：以列表的形式写入数据。
csv.DictReader：以字典的形式返回读取的数据。
csv.DictWriter：以字典的形式写入数据。
'''
imgData = torch.ones(1, 3, 600, 600)
imgLab = tansor_case.newTansor1(99)
imgLab=imgLab.unsqueeze(0)
i = 1
for imgPath, data2 in csv_reader:
    imgPath = imgPath.replace('\\', '/')
    imgPath = "E:/BaiduNetdiskDownload/baokemeng/" + imgPath
    k1 = pic_tensor1.imgToTensor(imgPath)
    k1 = k1.unsqueeze(0)

    imgData = torch.cat([imgData, k1], 0)
    # 这里我选择及时释放k1 但我还没有考虑后果
    del (k1)

    k2 = tansor_case.newTansor1(int(data2))
    k2 = k2.unsqueeze(0)
    imgLab=torch.cat([imgLab,k2],0)
    del(k2)

    print(i, ":", imgPath, data2)
    i = i + 1
# dir 方法可以看类内部方法和变量
# print(dir(csv_reader))
# print(len(csv_reader))

# 1000+的数据直接干碎我的内存
# 1167行数据
# print(csv_reader.line_num)

# E:/BaiduNetdiskDownload/baokemeng/pokemon/mewtwo/00000239.jpg 2
print(imgData.shape)
imgData = tansor_case.del_tansor_ele(imgData, 0)
imgLab=tansor_case.del_tansor_ele(imgLab,0)
#print(imgLab)
#print(imgData.shape,imgLab.shape)
