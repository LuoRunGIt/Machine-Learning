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