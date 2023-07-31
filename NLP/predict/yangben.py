import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# 设置显示⻛格
plt.style.use('fivethirtyeight')
#seaborn也是画图的
#文本数据分析
# 分别读取训练tsv和验证tsv
train_data = pd.read_csv("./train.tsv", sep="\t")
valid_data = pd.read_csv("./dev.tsv", sep="\t")
print(train_data)
# 获得训练数据标签数量分布
sns.countplot(x="label", data=train_data)
plt.title("train_data")
plt.show()
# 获取验证数据标签数量分布
sns.countplot(x="label", data=valid_data)
plt.title("valid_data")
plt.show()

#获取训练集和验证集的句⼦⻓度分布
# 在训练数据中添加新的句⼦⻓度列, 每个元素的值都是对应的句⼦列的⻓度
train_data["sentence_length"] = list(map(lambda x: len(x),train_data["sentence"]))
# 绘制句⼦⻓度列的数量分布图
sns.countplot(x="sentence_length", data=train_data)
# 主要关注count⻓度分布的纵坐标, 不需要绘制横坐标, 横坐标范围通过dist图进⾏查看
plt.xticks([])
plt.show()
# 绘制dist⻓度分布图
sns.distplot(train_data["sentence_length"])
# 主要关注dist⻓度分布横坐标, 不需要绘制纵坐标
plt.yticks([])
plt.show()
# 在验证数据中添加新的句⼦⻓度列, 每个元素的值都是对应的句⼦列的⻓度
valid_data["sentence_length"] = list(map(lambda x: len(x),
valid_data["sentence"]))
# 绘制句⼦⻓度列的数量分布图
sns.countplot(x="sentence_length", data=valid_data)
# 主要关注count⻓度分布的纵坐标, 不需要绘制横坐标, 横坐标范围通过dist图进⾏查看
plt.xticks([])
plt.show()

# 绘制训练集⻓度分布的散点图
sns.stripplot(y='sentence_length',x='label',data=train_data)
plt.show()
# 绘制验证集⻓度分布的散点图
sns.stripplot(y='sentence_length',x='label',data=valid_data)
plt.show()

# 导⼊jieba⽤于分词
# 导⼊chain⽅法⽤于扁平化列表
import jieba
from itertools import chain
# 进⾏训练集的句⼦进⾏分词, 并统计出不同词汇的总数
train_vocab = set(chain(*map(lambda x: jieba.lcut(x),
train_data["sentence"])))
print("训练集共包含不同词汇总数为：", len(train_vocab))
# 进⾏验证集的句⼦进⾏分词, 并统计出不同词汇的总数
valid_vocab = set(chain(*map(lambda x: jieba.lcut(x),
valid_data["sentence"])))
print("训练集共包含不同词汇总数为：", len(valid_vocab))