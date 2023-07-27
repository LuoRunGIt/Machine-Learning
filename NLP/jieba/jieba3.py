import jieba.posseg as pesg
#词性标注
data=pesg.lcut("我爱北京天安门")
print(data)