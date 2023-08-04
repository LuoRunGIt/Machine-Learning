'''
如果参数1形状是(b × n × m), 参数2形状是(b × m × p), 则输出为(b × n × p)

'''
import torch
mat1=torch.randn(10,4,5)
mat2=torch.randn(10,5,6)

res=torch.bmm(mat1,mat2)
print(res.size())