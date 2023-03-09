import numpy as np
A = np.array([[1],[2]])
B = np.array([[1,2,4],[1,4,5]])
X=A*B
Y=A.T@B
print(X)
print(Y)