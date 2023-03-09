a = [1, 2, 3]
b = [4, 5, 6]
x = zip(a, b)
print(list(x))
a1, b1 = zip(*zip(a, b))
#a2, b2 = zip(*zip(x))
print(a1, b1)
#print(a2, b2)
c = [(7, 8, 9),(10,11,12)]
y = zip(x, c)
print(list(y))
