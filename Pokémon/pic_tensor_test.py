import pic_tensor1
import torch
path1="E:/BaiduNetdiskDownload/baokemeng/pokemon/squirtle/00000154.jpg"
path2="E:/BaiduNetdiskDownload/baokemeng/pokemon/mewtwo/00000239.jpg"

k1=pic_tensor1.imgToTensor(path1)
k2=pic_tensor1.imgToTensor(path2)
print(k1.shape)
print(k2.shape)
#0 按行拼接，1按列拼接
#torch.Size([6, 1200, 1200])
data =torch.cat([k1,k2],0)
print(data.shape)
#torch.Size([2, 3, 1200, 1200])
c=torch.stack([k1,k2],0)
print(c.shape)