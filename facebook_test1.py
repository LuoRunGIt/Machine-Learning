import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV  # 数据集划分
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# import importlib,sys
# importlib.reload(sys)
# sys.setdefaultencoding('utf8')
facebook = pd.read_csv("E:\\BaiduNetdiskDownload\\课程资料\\05-机器学习\\02-机器学习代码\\chapter08\\data\\FBlocation\\train.csv")

# print(facebook.head(), facebook.shape)

'''
    目标1
    1.缩小数据范围，实际工作中不能随意缩减
    2.时间戳转时间 单位s
    3.签到位置少于n个用户的删除
    accuracy 误差值
'''
'''
    具体步骤：
# 1.获取数据集
# 2.基本数据处理
# 2.1 缩⼩数据范围
# 2.2 选择时间特征
# 2.3 去掉签到较少的地⽅
# 2.4 确定特征值和⽬标值
# 2.5 分割数据集
# 3.特征⼯程 -- 特征预处理(标准化)
# 4.机器学习 -- knn+cv
# 5.模型评估
'''

# 2.数据处理
# 2.1缩小范围 query查询
facebook_data = facebook.query("x>2.0&x<2.5&y>2.0&y<2.5")
# print(facebook_data.loc[:,"time"])
print(facebook_data.shape)
# 选择时间特征
time = pd.to_datetime(facebook_data.loc[:, "time"], unit="s")
time = pd.DatetimeIndex(time)
# print(time)
facebook_data.insert(loc=5, column="day", value=time.day)
facebook_data.insert(loc=6, column="hour", value=time.hour)
facebook_data.insert(loc=7, column="weekday", value=time.weekday)
# facebook_data.loc[:,"hour"] = time.hour
# facebook_data.loc[:,"weekday"] = time.weekday
# print(facebook_data.head())
# 2.3 去掉签到较少的地⽅

place_count = facebook_data.groupby("place_id").count()
# print(type(place_count))这里调试效果很好，一开始显示method结果是少个括号
place_count = place_count[place_count["row_id"] > 3]
facebook_data = facebook_data[facebook_data["place_id"].isin(place_count.index)]

# 划分特征值和目标
x = facebook_data[["x", "y", "accuracy", "day", "hour", "weekday"]]
y = facebook_data["place_id"]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22, test_size=0.3)

# 标准化
transfer = StandardScaler()
# 标准化训练集
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 训练
# 交叉验证
# 估计器
estimator = KNeighborsClassifier()
# n_neighbors
param_grid1 = {"n_neighbors": [1, 3, 5, 7, 9]}
# -1表示所有所有cup
# 这里有一个utf-8编码的问题n_jobs
estimator = GridSearchCV(estimator, param_grid=param_grid1, cv=3)
estimator.fit(x_train, y_train)

# 模型验证
score = estimator.score(x_test, y_test)
print("最后预测的准确率为:\n", score)
y_predict = estimator.predict(x_test)
print("最后的预测值为:\n", y_predict)
print("预测值和真实值的对⽐情况:\n", y_predict == y_test)
# 5.2 使⽤交叉验证后的评估⽅式
print("在交叉验证中验证的最好结果:\n", estimator.best_score_)
print("最好的参数模型:\n", estimator.best_estimator_)
print("每次交叉验证后的验证集准确率结果和训练集准确率结果:\n", estimator.cv_results_)
