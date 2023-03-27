import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
#import graphviz

#data_train=pd.read_csv("E:\\BaiduNetdiskDownload\\课程资料\\05-机器学习\\02-机器学习代码\\chapter10\\data\\otto\\train.csv")
#data_test=pd.read_csv("E:\\BaiduNetdiskDownload\\课程资料\\05-机器学习\\02-机器学习代码\\chapter10\\data\\otto\\test.csv")
#train 中有一列target表示分类结果
#print(data_train.info())
#print(data_test.info())


#不过案例用的是另一个
titan=pd.read_csv("E:\\BaiduNetdiskDownload\\课程资料\\05-机器学习\\02-机器学习代码\\chapter10\\data\\otto\\titanic.csv")
x = titan[["pclass", "age", "sex"]]
y = titan["survived"]
#缺失值处理
k=x['age'].mean()
print(x.head())
print(type(x))
print(x.loc[:,'age'])

print(k)
#x.loc[:,'age'].fillna(k,inplace=True)
#x.loc[x.age=='Nan=N', 'age'] =k
#x.fillna({"age":k})
#x.dropna()
x.loc[:,'age'].fillna(x['age'].mean(),inplace=True)
print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22,train_size=0.2)
#print(x_train)
#数据标准化
#特征中出现类别符号，需要进⾏one-hot编码处理(DictVectorizer)
#x.to_dict(orient="records") 需要将数组特征转换成字典数据
transfer = DictVectorizer(sparse=False)
x_train = transfer.fit_transform(x_train.to_dict(orient="records"))
x_test = transfer.fit_transform(x_test.to_dict(orient="records"))
print(x_train)

estimator = DecisionTreeClassifier(criterion="entropy", max_depth=5)
estimator.fit(x_train, y_train)
# 5.模型评估
score=estimator.score(x_test, y_test)
test_score=estimator.predict(x_test)
'''
print(score,test_score)
export_graphviz(estimator, out_file="./data/tree.dot", feature_names=['age', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', '女性', '男性'])
with open("./data/tree.dot",'r', encoding='UTF-8') as f:
    dot_graph = f.read()
dot=graphviz.Source(dot_graph)
dot.view()
'''