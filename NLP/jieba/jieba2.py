import jieba
'''引入用户自定义词典 '''
jieba.add_word("大熊猫")
jieba.add_word("辨识度")
jieba.load_userdict('./userdict.txt')
content="和花，雌性大熊猫，谱系号1237，2020年7月4日与双胞胎弟弟和叶出生在成都大熊猫繁育研究基地月亮产房，初生体重200g，为辨识度和颜值都很高的一只熊猫。"
myobj1=jieba.cut(content,cut_all=False)

print([i for i in jieba.cut(content)])
print("myobj1-->",myobj1)