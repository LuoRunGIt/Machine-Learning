import numpy as np
import torch

a=torch.ones(5)
print(a)

b=a.numpy()
a.add_(1)

print(b)#两者直接共享底层内存空间

#numpy转tansor
a1 = np.ones(5)
b1 = torch.from_numpy(a1)
np.add(a1, 1, out=a1)
print(a1)
print(b1)