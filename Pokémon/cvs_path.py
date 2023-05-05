'''
这里将 文件路径对应的图片读取出来
'''
import csv
csv_reader = csv.reader(open("E:\\BaiduNetdiskDownload\\baokemeng\\pokemon\\images.csv"))

'''
csv.reader：以列表的形式返回读取的数据。
csv.writer：以列表的形式写入数据。
csv.DictReader：以字典的形式返回读取的数据。
csv.DictWriter：以字典的形式写入数据。
'''
for data1,data2 in csv_reader:
	data1=data1.replace('\\','/')
	data1="E:/BaiduNetdiskDownload/baokemeng/"+data1

	print(data1,data2)
#dir 方法可以看类内部方法和变量
#print(dir(csv_reader))
#print(len(csv_reader))

# 1167行数据
#print(csv_reader.line_num)

#E:/BaiduNetdiskDownload/baokemeng/pokemon/mewtwo/00000239.jpg 2