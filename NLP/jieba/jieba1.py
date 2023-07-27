import jieba

'''jieba分词器练习
精确模式
 
'''

#精确模式
content="和花，雌性大熊猫，谱系号1237，2020年7月4日与双胞胎弟弟和叶出生在成都大熊猫繁育研究基地月亮产房，初生体重200g，为辨识度和颜值都很高的一只熊猫。"
myobj1=jieba.cut(sentence=content,cut_all=False)
print("myobj1-->",myobj1)

mydata1=jieba.lcut(sentence=content,cut_all=False)
print("mydata1-->",mydata1)

#全词模式
mydata2=jieba.lcut(sentence=content,cut_all=True)
print("mydata2-->",mydata1)

#搜索引擎模式，在精确模式基础上对长词做切分，适合搜索
myobj2=jieba.cut_for_search(sentence=content)
print("myobj2-->",myobj2)
mydata3=jieba.lcut_for_search(sentence=content)
print("mydata3-->",mydata3)