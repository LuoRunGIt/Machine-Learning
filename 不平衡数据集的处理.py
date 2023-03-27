from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler

#随机过采样随机过采样是在少数类 中随机选择⼀些样本，然后通过复制所选择的样本⽣成样本集 ，将它们添加到 中来扩⼤原始数据集从⽽得到新的少数类集合 。新的数据集
# pip install imbalanced-learn
from collections import Counter
X, y = make_classification(n_samples=5000,
                           n_features=2,  # 特征个数= n_informative（） + n_redun
                           n_informative=2,  # 多信息特征的个数
                           n_redundant=0,  # 冗余信息，informative特征的随机线性组
                           n_repeated=0,  # 重复信息，随机提取n_informative和n_red
                           n_classes=3,  # 分类类别
                           n_clusters_per_class=1,  # 某⼀个类别是由⼏个cluster构成
                           weights=[0.01, 0.05, 0.94],  # 列表类型，权重⽐
                           random_state=0)
k=Counter(y)
print(X.shape,y.shape,k)
#plt.scatter(X[:,0],X[:,1],c=y)
#plt.show()
#随机过采样
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X, y)
k1=Counter(y_resampled)
print(k1)
plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c=y_resampled)
plt.show()

from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE().fit_resample(X, y)
Counter(y_resampled)
# 采样后样本结果
# [(0, 4674), (1, 4674), (2, 4674)]
# 数据集可视化
plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c=y_resampled)
plt.show()

# 随机⽋采样
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = rus.fit_resample(X, y)
Counter(y_resampled)
# 采样后结果
#[(0, 64), (1, 64), (2, 64)]
# 数据集可视化
plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c=y_resampled)
plt.show()