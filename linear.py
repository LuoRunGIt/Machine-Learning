from sklearn.linear_model import LinearRegression
x = [[80, 86],
[82, 80],
[85, 78],
[90, 90],
[86, 82],
[82, 90],
[78, 80],
[92, 94]]
y = [84.2, 80.6, 80.1, 90, 83.2, 87.6, 79.4, 93.4]
# 实例化API
estimator = LinearRegression()
# 使⽤fit⽅法进⾏训练
estimator.fit(x,y)
print(estimator.coef_)
estimator.predict([[100, 80]])