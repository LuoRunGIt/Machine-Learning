seasons=['Spring', 'Summer', 'Fall', 'Winter']
k=list(enumerate(seasons))
p=list(enumerate(seasons,start=1))
print(k)
print(p)

seq=["one","two","three"]
for i,element in enumerate(seq):
    print(i,element)