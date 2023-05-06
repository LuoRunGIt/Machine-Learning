import pic_tensor1
import tansor_case
import torch
path1="E:/BaiduNetdiskDownload/baokemeng/pokemon/squirtle/00000154.jpg"
path2="E:/BaiduNetdiskDownload/baokemeng/pokemon/mewtwo/00000239.jpg"

k1=pic_tensor1.imgToTensor(path1)
k2=pic_tensor1.imgToTensor(path2)
k3=pic_tensor1.imgToTensor(path1)
print(k1.shape)
print(k2.shape)
#0 按行拼接，1按列拼接
#torch.Size([6, 1200, 1200])
data =torch.cat([k1,k2],0)
print(data.shape)
#torch.Size([2, 3, 1200, 1200])
c=torch.stack([k1,k2],0)
del(k2)
print(k1)
print(c)
print(c.shape)
#这里没有办法合并咋办，我升维度
k3=k3.unsqueeze(0)
print(k3.shape)
data1 =torch.cat([c,k3],0)
print(data1.shape)

k4=tansor_case.del_tansor_ele(data1,0)
print(k4.shape)