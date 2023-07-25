#iter为迭代器
import ctypes

list1 = [8,2,3,4]
for i in iter(list1):
    print(i)


list2=[[1,2],[2,3]]
for i in iter(list2):
    print(i)
k=iter(list2)
print(k)
for i in k:
    print(i)
'''

next(k)
#实际上next方法需要类自身实现
print(k)
for i in k:
    print(i)
'''