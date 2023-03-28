from sklearn.feature_selection import VarianceThreshold
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
def variance_demo():
    """
    删除低⽅差特征——特征选择
    :return: None
    """
    data = pd.read_csv("E:\\BaiduNetdiskDownload\\课程资料\\05-机器学习\\02-机器学习代码\\chapter12\\data\\factor_returns.csv")
    print(data.shape)
    # 1、实例化⼀个转换器类
    transfer = VarianceThreshold(threshold=1)
    # 2、调⽤fit_transform
    data = transfer.fit_transform(data.iloc[:, 1:10])
    print("删除低⽅差特征的结果：\n", data)
    print("形状：\n", data.shape)
    return None


variance_demo()


def pea_demo():
 """
 皮尔逊相关系数
 :return:
 """
 # 准备数据
 x1 = [12.5, 15.3, 23.2, 26.4, 33.5, 34.4, 39.4, 45.2, 55.4, 60.9]
 x2 = [21.2, 23.9, 32.9, 34.1, 42.5, 43.2, 49.0, 52.8, 59.4, 63.5]

 # 判断
 ret = pearsonr(x1, x2)
 print("皮尔逊相关系数的结果是:\n", ret)

pea_demo()
def spea_demo():
 """
 斯皮尔曼相关系数
 :return:
 """
 # 准备数据
 x1 = [12.5, 15.3, 23.2, 26.4, 33.5, 34.4, 39.4, 45.2, 55.4, 60.9]
 x2 = [21.2, 23.9, 32.9, 34.1, 42.5, 43.2, 49.0, 52.8, 59.4, 63.5]

 # 判断
 ret = spearmanr(x1, x2)
 print("斯皮尔曼相关系数的结果是:\n", ret)

spea_demo()
def pca_demo():
 """
 pca降维
 :return:
 """
 data = [[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]]

 # pca小数保留百分比
 transfer = PCA(n_components=0.9)
 trans_data = transfer.fit_transform(data)
 print("保留0.9的数据最后维度为:\n", trans_data)#结果竟然是2维

 # pca小数保留百分比
 transfer = PCA(n_components=3)
 trans_data = transfer.fit_transform(data)
 print("保留三列数据:\n", trans_data)

pca_demo()