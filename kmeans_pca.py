import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score  # 评分

# 用户与商品之间的关系
order_product = pd.read_csv("E:\\BaiduNetdiskDownload\\课程资料\\05-机器学习"
                            "\\02-机器学习代码\\chapter12\\data\instacart\\order_products__prior.csv")
products = pd.read_csv("E:\\BaiduNetdiskDownload\\课程资料\\05-机器学习\\02-机器学习代码\\chapter12\\data\\instacart\\products.csv")
orders = pd.read_csv("E:\\BaiduNetdiskDownload\\课程资料\\05-机器学习\\02-机器学习代码\\chapter12\\data\\instacart\\orders.csv")
aisles = pd.read_csv("E:\\BaiduNetdiskDownload\\课程资料\\05-机器学习\\02-机器学习代码\\chapter12\\data\\instacart\\aisles.csv")
table1 = pd.merge(order_product, products, on=["product_id", "product_id"])
table2 = pd.merge(table1, orders, on=["order_id", "order_id"])
table = pd.merge(table2, aisles, on=["aisle_id", "aisle_id"])
#交叉表合并
table = pd.crosstab(table["user_id"], table["aisle"])
table = table[:100]#源为1000
transfer = PCA(n_components=0.9)
data = transfer.fit_transform(table)
print(data.shape)
# estimator = KMeans(n_clusters=8, random_state=22)
estimator = KMeans(n_clusters=5)
y_predict = estimator.fit_predict(data)
k = silhouette_score(data, y_predict)
print(k)
# 1000，22
