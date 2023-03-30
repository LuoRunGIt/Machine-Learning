import pandas as pd
import numpy as np
import matplotlib.pyplot as  plt
from sklearn import  svm
from sklearn.model_selection import train_test_split
import time
from sklearn.decomposition import PCA
train=pd.read_csv("E:\\BaiduNetdiskDownload\\课程资料\\05-机器学习\\02-机器学习代码\\chapter14\data\\train.csv")
print(train.shape)
#785列数据太多需要降维
print(train)
#iloc[:,1:] 正确写法
train_image=train.loc[:,train.columns!='label']
print(train_image)

train_label=train.loc[:,"label"]
#
print(train_label)

#特征值处理
#归一化
train_image=train_image/255
train_label=train_label.values
print(train_label)
x_train, x_val, y_train, y_val = train_test_split(train_image, train_label, train_size = 0.8, random_state=0)


#看图
num=train_image.loc[1].values.reshape(28,28)
plt.imshow(num)
plt.axis("off")#这样写就没有坐标轴了，axis表示轴
plt.show()


# 传递多个n_components,寻找合理的n_components:
#在线性空间中生成均匀步长的数列
n_s = np.linspace(0.70, 0.85, num=5)
accuracy = []
print(n_s)


# 多次使用pca,确定最后的最优模型

def n_components_analysis(n, x_train, y_train, x_val, y_val):
    # 记录开始时间
    start = time.time()

    # pca降维实现,n表示保留n%*100的特征
    pca = PCA(n_components=n)#实例化
    print("特征降维,传递的参数为:{}".format(n))
    pca.fit(x_train)

    # 在训练集和测试集进行降维
    x_train_pca = pca.transform(x_train)
    x_val_pca = pca.transform(x_val)

    # 利用svc进行训练
    print("开始使用svc进行训练")
    ss = svm.SVC()
    ss.fit(x_train_pca, y_train)

    # 获取accuracy结果，accuracy准确性
    accuracy = ss.score(x_val_pca, y_val)

    # 记录结束时间
    end = time.time()
    print("准确率是:{}, 消耗时间是:{}s".format(accuracy, int(end - start)))

    return accuracy
for n in n_s:
    tmp = n_components_analysis(n, x_train, y_train, x_val, y_val)
    accuracy.append(tmp)
# 准确率可视化展示
plt.plot(n_s, np.array(accuracy), "r")
plt.show()
#最后0.85 训练的极慢
'''
pca = PCA(n_components=0.80)

pca.fit(x_train)
pca.n_components_
x_train_pca = pca.transform(x_train)
x_val_pca = pca.transform(x_val)
print(x_train_pca.shape, x_val_pca.shape)

ss1 = svm.SVC()

ss1.fit(x_train_pca, y_train)
#带入测试集进行打分
ss1.score(x_val_pca, y_val)

'''
#注意不同的机器学习算法中的score是不一样的