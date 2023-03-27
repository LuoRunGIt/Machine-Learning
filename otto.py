# otto平台数据，随机森林案例
# 区分产品类别
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler  # 随机欠采样的数据集
from sklearn.preprocessing import LabelEncoder  # 把标签值进行数值化
from sklearn.model_selection import train_test_split  # 数据集划分
from sklearn.ensemble import RandomForestClassifier  # 随机森林
from sklearn.metrics import log_loss  # 模型评估
from sklearn.preprocessing import OneHotEncoder  # 转one-hot编码

# 读取数据
data = pd.read_csv("E:\\BaiduNetdiskDownload\\课程资料\\05-机器学习\\02-机器学习代码\\chapter11\\data\\otto\\train.csv")
print(data.head(), data.tail(), data.shape)
# 61878个样本，95列，第一列为序号，2-94列为特征 95列为分类

# 数据可视化
sns.countplot(x=data.target)
plt.show()
# 数据处理

# 这种获取方式显然是不行的，只能获得target 1 2的数据
# new1_data = data[:10000]
# print(new1_data.shape)


# 使用随机欠采样的方式进行数据集获取
y = data["target"]
x = data.drop(["id", "target"], axis=1)
rus = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = rus.fit_resample(x, y)

print(X_resampled.shape, y_resampled.shape)
sns.countplot(x=y_resampled)
plt.show()

# 标签值数值化
le = LabelEncoder()
y_resampled = le.fit_transform(y_resampled)
print(y_resampled)

# 数据集划分
x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2)

# 这里不需要标准化transfer.fit_transform

# 模型训练 用袋外数据来测试则oob_score这个参数为True
rf = RandomForestClassifier(oob_score=True)  # 有袋外数据了，为何还要测试集呢
rf.fit(x_train, y_train)

y_pre = rf.predict(x_test)
print("预测结果", y_pre)
k = rf.score(x_test, y_test)
print("测试集结果", k, "包外数据结果", rf.oob_score_)
sns.countplot(x=y_pre)
plt.show()

### 模型调优

y_test.reshape(-1, 1)
y_pre.reshape(-1, 1)
one_hot = OneHotEncoder(sparse_output=False)

y_test1 = one_hot.fit_transform(y_test.reshape(-1, 1))
y_pre1 = one_hot.fit_transform(y_pre.reshape(-1, 1))
log_loss(y_test1, y_pre1, eps=1e-15, normalize=True)
# 改变预测值的输出模式,让输出结果为百分占比,降低logloss值
y_pre_proba = rf.predict_proba(x_test)
log_loss(y_test1, y_pre_proba, eps=1e-15, normalize=True)  # normalize 是否标准化

## 参数调优
# 确定n_estimators的取值范围，即树的个数
tuned_parameters = range(10, 200, 10)

# 创建添加accuracy的一个numpy
accuracy_t = np.zeros(len(tuned_parameters))

# 创建添加error的一个numpy
error_t = np.zeros(len(tuned_parameters))

# 调优过程实现
for j, one_parameter in enumerate(tuned_parameters):
    rf2 = RandomForestClassifier(n_estimators=one_parameter,
                                 max_depth=10,
                                 max_features=10,
                                 min_samples_leaf=10,
                                 oob_score=True,
                                 random_state=0,
                                 n_jobs=-1)

    rf2.fit(x_train, y_train)

    # 输出accuracy
    accuracy_t[j] = rf2.oob_score_

    # 输出log_loss
    y_pre = rf2.predict_proba(x_test)
    error_t[j] = log_loss(y_test, y_pre, eps=1e-15, normalize=True)

    print(error_t)
# 优化结果过程可视化
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 4), dpi=100)

axes[0].plot(tuned_parameters, error_t)
axes[1].plot(tuned_parameters, accuracy_t)
axes[0].set_xlabel("n_estimators")
axes[0].set_ylabel("error_t")
axes[1].set_xlabel("n_estimators")
axes[1].set_ylabel("accuracy_t")

axes[0].grid(True)
axes[1].grid(True)

plt.show()

# 确定max_features的取值范围
tuned_parameters = range(5, 40, 5)

# 创建添加accuracy的一个numpy
accuracy_t = np.zeros(len(tuned_parameters))

# 创建添加error的一个numpy
error_t = np.zeros(len(tuned_parameters))

# 调优过程实现
for j, one_parameter in enumerate(tuned_parameters):
    rf2 = RandomForestClassifier(n_estimators=175,
                                 max_depth=10,
                                 max_features=one_parameter,
                                 min_samples_leaf=10,
                                 oob_score=True,
                                 random_state=0,
                                 n_jobs=-1)

    rf2.fit(x_train, y_train)

    # 输出accuracy
    accuracy_t[j] = rf2.oob_score_

    # 输出log_loss
    y_pre = rf2.predict_proba(x_test)
    error_t[j] = log_loss(y_test, y_pre, eps=1e-15, normalize=True)

    print(error_t)

# 优化结果过程可视化
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 4), dpi=100)

axes[0].plot(tuned_parameters, error_t)
axes[1].plot(tuned_parameters, accuracy_t)

axes[0].set_xlabel("max_features")
axes[0].set_ylabel("error_t")
axes[1].set_xlabel("max_features")
axes[1].set_ylabel("accuracy_t")

axes[0].grid(True)
axes[1].grid(True)

plt.show()

# 确定max_depth的取值范围
tuned_parameters = range(10, 100, 10)

# 创建添加accuracy的一个numpy
accuracy_t = np.zeros(len(tuned_parameters))

# 创建添加error的一个numpy
error_t = np.zeros(len(tuned_parameters))

# 调优过程实现
for j, one_parameter in enumerate(tuned_parameters):
    rf2 = RandomForestClassifier(n_estimators=175,
                                 max_depth=one_parameter,
                                 max_features=15,
                                 min_samples_leaf=10,
                                 oob_score=True,
                                 random_state=0,
                                 n_jobs=-1)

    rf2.fit(x_train, y_train)

    # 输出accuracy
    accuracy_t[j] = rf2.oob_score_

    # 输出log_loss
    y_pre = rf2.predict_proba(x_test)
    error_t[j] = log_loss(y_test, y_pre, eps=1e-15, normalize=True)

    print(error_t)

# 优化结果过程可视化
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 4), dpi=100)

axes[0].plot(tuned_parameters, error_t)
axes[1].plot(tuned_parameters, accuracy_t)

axes[0].set_xlabel("max_depth")
axes[0].set_ylabel("error_t")
axes[1].set_xlabel("max_depth")
axes[1].set_ylabel("accuracy_t")

axes[0].grid(True)
axes[1].grid(True)

plt.show()

# 确定min_sample_leaf的取值范围
tuned_parameters = range(1, 10, 2)

# 创建添加accuracy的一个numpy
accuracy_t = np.zeros(len(tuned_parameters))

# 创建添加error的一个numpy
error_t = np.zeros(len(tuned_parameters))

# 调优过程实现
for j, one_parameter in enumerate(tuned_parameters):
    rf2 = RandomForestClassifier(n_estimators=175,
                                 max_depth=30,
                                 max_features=15,
                                 min_samples_leaf=one_parameter,
                                 oob_score=True,
                                 random_state=0,
                                 n_jobs=-1)

    rf2.fit(x_train, y_train)

    # 输出accuracy
    accuracy_t[j] = rf2.oob_score_  # 准确度

    # 输出log_loss
    y_pre = rf2.predict_proba(x_test)
    error_t[j] = log_loss(y_test, y_pre, eps=1e-15, normalize=True)

    print(error_t)

# 优化结果过程可视化
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 4), dpi=100)

axes[0].plot(tuned_parameters, error_t)
axes[1].plot(tuned_parameters, accuracy_t)

axes[0].set_xlabel("min_sample_leaf")
axes[0].set_ylabel("error_t")
axes[1].set_xlabel("min_sample_leaf")
axes[1].set_ylabel("accuracy_t")

axes[0].grid(True)
axes[1].grid(True)

plt.show()

###优化结果带入
rf3 = RandomForestClassifier(n_estimators=175, max_depth=30, max_features=40, min_samples_leaf=1,
                             oob_score=True, random_state=40, n_jobs=-1)

rf3.fit(x_train, y_train)

score1 = rf3.score(x_test, y_test)
print(score1, rf3.oob_score_)
y_pre_proba1 = rf3.predict_proba(x_test)
print(y_pre_proba1)
k = log_loss(y_test, y_pre_proba1)
print(k)

##数据结果上交
test_data = pd.read_csv("E:\\BaiduNetdiskDownload\\课程资料\\05-机器学习\\02-机器学习代码\\chapter11\\data\\otto\\test.csv")
test_data_drop_id = test_data.drop(["id"], axis=1)
y_pre_test = rf3.predict_proba(test_data_drop_id)
result_data = pd.DataFrame(y_pre_test, columns=["Class_" + str(i) for i in range(1, 10)])
result_data.insert(loc=0, column="id", value=test_data.id)
result_data.to_csv("E:\\BaiduNetdiskDownload\\课程资料\\05-机器学习\\02-机器学习代码\\chapter11\\data\\otto\\submission1.csv",
                   index=False)
