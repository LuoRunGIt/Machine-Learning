import torch


def del_tansor_ele(arr, index=0):
    if index <= 0:
        return arr[1:index + 1]
    if arr == None:
        print("not tansor")
        return None
    arr1 = arr[0:index]
    arr2 = arr[index + 1:]
    return torch.cat((arr1, arr2), 0)


# https://blog.csdn.net/qq_43391414/article/details/120468225
# 创建不同类型的tansor 以节约空间
def newTansor1(val):
    return torch.IntTensor([val])
