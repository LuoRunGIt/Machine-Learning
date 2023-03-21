import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def minmax_demo():
    """
    归⼀化演示
    :return: None
    """
    data = pd.read_csv("E:\\BaiduNetdiskDownload\\课程资料\\05-机器学习\\02-机器学习代码\\chapter07\\02-代码\\data\\dating.txt")
    print(data)
    # 1、实例化⼀个转换器类
    transfer = MinMaxScaler(feature_range=(2, 3))#归一化不一定在0，1之间
    # 2、调⽤fit_transform
    data = transfer.fit_transform(data[['milage', 'Liters', 'Consumtime']])
    print("最⼩值最⼤值归⼀化处理的结果：\n", data)
    return None

minmax_demo()
