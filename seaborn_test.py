import seaborn as sns
import numpy as np

dataset = sns.load_dataset("iris", data_home="C:\\Users\\骆润\\Desktop\\seaborn-data")
# print(dataset)

print(type(dataset))

'''
def all_house(arr):
    key = np.unique(arr)  # 去重
    result = {}
    for k in key:
        mask = (arr == k)
        # print(mask)
        arr_new = arr[mask]
        print(arr_new)
        v = arr_new.size
        result[k] = v
    return result
'''
