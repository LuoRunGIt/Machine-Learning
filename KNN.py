from sklearn.neighbors import KNeighborsClassifier

# k默认是5
x = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]
estimator = KNeighborsClassifier(n_neighbors=2)
estimator.fit(x, y)  # y表示类别
estimator.predict([[-1]])
estimator = KNeighborsClassifier(n_neighbors=2)
estimator.fit(x, y)  # y表示类别
a = estimator.predict([[-1.1]])
print(a)

