# eval函数直接返回字符串中的数据类型
str1 = "1"
print(type(eval(str1)))
a = b = 1
print(id(a), id(b))
a = a + 1
print(id(a), id(b))
