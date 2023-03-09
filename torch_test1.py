import torch
import os
import pandas as pd

x = torch.arange(16)
x = x.reshape((4, 4))
print(x)
print(x[-1])  # 最后一行
print(x[0])
print(x[0:1])  # 1，2行，左闭右开
print(x[1:3])

os.makedirs(os.path.join("./", "data"), exist_ok=True)
data_file = os.path.join('./', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:  # with语法会在使用完毕后释放资源，比如这里的文件流
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
    f.write('3,NA,240000\n')
data = pd.read_csv('./data/house_tiny.csv')
print(data)

# 处理缺失值，插值法或者删除法
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())  # fillna表示缺失值填充，这里填充的是均值
print(inputs)
inputs = pd.get_dummies(inputs, dummy_na=True)  # 将字符类，转化为多个列，然后用1，0 填充；可量化但是增加了维度
print(inputs)

x,y=torch.tensor(inputs.values),torch.tensor(outputs.values)
print(x,y)
print(len(x),len(y))
#pandas可以和张量兼容
k=torch.arange(4)
print(len(k))