import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
print(data.shape)
print(data.data.shape, target.shape)

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
# print(x_train)

# 标准化
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 训练
# 交叉
# 有2种方式，一是正规方程，二是梯度下降
# "estimator"中文翻译 n. 估计者
# estimator = LinearRegression()#正规方程
# 梯度下降法
# estimator = SGDRegressor(max_iter=1000,learning_rate="constant",eta0=0.1)#最大迭代次数,这里设置学习率是固定值，效果不佳
estimator = Ridge(alpha=1)#岭回归
estimator.fit(x_train, y_train)
print("偏置", estimator.intercept_)
print("系数", estimator.coef_)
# 模型评估
y_pre = estimator.predict(x_test)
print("预测值为:\n", y_pre)
error = mean_squared_error(y_test, y_pre)
print("误差为:\n", error)

fig = plt.figure(figsize=(12, 6))
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体

plt.plot(range(y_test.shape[0]), y_test, color='blue', linewidth=1.5, linestyle='-')
plt.plot(range(y_test.shape[0]), y_pre, color='red', linewidth=1.5, linestyle='-.')
plt.legend(["原始值", "预测试"])
plt.show()

#模型保存
joblib.dump(estimator, './data/linghuigui.pkl')
# 加载
# estimator = joblib.load('test.pkl')
#模型大时可以边训练边保存