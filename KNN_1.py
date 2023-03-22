from sklearn.datasets import load_iris
# 获取鸢尾花数据集
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV  # 数据集划分
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# 设置字体为楷体
plt.rcParams['font.sans-serif'] = ['KaiTi']

# 1. 导入数据，获取数据集
iris = load_iris()
print("鸢尾花数据集的返回值：\n", iris)
# 返回值是⼀个继承⾃字典的Bench
print("鸢尾花的特征值:\n", iris["data"])
print("鸢尾花的⽬标值：\n", iris.target)
print("鸢尾花特征的名字：\n", iris.feature_names)
print("鸢尾花⽬标值的名字：\n", iris.target_names)
# print("鸢尾花的描述：\n", iris.DESCR)

print(iris.keys())
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])

'''
data：特征数据数组，是 [n_samples * n_features] 的⼆维
numpy.ndarray 数组
target：标签数组，是 n_samples 的⼀维 numpy.ndarray 数组
DESCR：数据描述
feature_names：特征名,新闻数据，⼿写数字、回归数据集没有
target_names：标签名
'''
print(iris.data_module)  # 数据模块sklearn.datasets.data
# print(iris.frame)#框架这个数据集没有用到
# 数据集，列
iris_d = pd.DataFrame(iris['data'], columns=['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width'])
# 这里等于是新增一列
iris_d['Species'] = iris.target


# print(iris_d)

# 定义一个函数实现变量两两组合画图
def plot_iris(iris, col1, col2):
    # seaborn画图lmplot是用来绘制回归图的,hue简单理解：按照hue指定的特征或标签的类别的颜色种类进行区分
    sns.lmplot(x=col1, y=col2, data=iris, hue="Species", fit_reg=False)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title("鸢尾花种类分布图")
    plt.show()


plot_iris(iris_d, 'Petal_Length', 'Sepal_Length')

# 2 数据基本处理
'''  ------------------------------数据集划分----------------'''
'''x_train 特征值训练集 x_test 特征值测试集
    y_train 目标值训练集 y_test 目标值测试集
    random_state 随机数种子
    test_size 测试集占比，默认30%
'''
print(iris.data.shape)
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=2, test_size=0.3)
print(x_train.shape, x_test.shape)

# 3 特征工程数据标准化
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)  # 如果x_test不标准化，预测值就不准确，这里沿用了训练集的均值和方差
# 如果我用fit_transform 那么用的是训练集的均值和方差
# https://blog.csdn.net/weixin_38278334/article/details/82971752
# 4训练模型
# 选择要调整的超参数
param_dict = {"n_neighbors": [1, 3, 5, 7]}
# 交叉验证

# estimator = KNeighborsClassifier(n_neighbors=9)
# 交叉验证时不需要参数9
estimator = KNeighborsClassifier()
estimator = GridSearchCV(estimator, param_grid=param_dict, cv=10)  # 网格搜索

estimator.fit(x_train, y_train)  # y表示类别
# 5评估模型
y_predict = estimator.predict(x_test)
print("预测结果为:\n", y_predict)
print("⽐对真实值和预测值：\n", y_predict == y_test)
# 可以自己算
score = estimator.score(x_test, y_test)
print("准确率为：\n", score)

# 这里是交叉验证后的属性
print("在交叉验证中验证的最好结果：\n", estimator.best_score_)
print("最好的参数模型：\n", estimator.best_estimator_)
print("每次交叉验证后的准确率结果：\n", estimator.cv_results_)
