# @Time    : 2021/12/13 20:15
# @Author  : 南黎
# @FileName: 1.创建数据库.py
import pymysql
import rsa1

# 创建与数据库的连接
# 创建连接
# host      主机名字，本机一般都是localhost
# user      用户名
# password  用户密码
# charset   数据库编码

#这里简单加密了一下
f=open('./crypto')
crypto = f.readlines()  # 直接将文件中按行读到list里，效果与方法2一样
f.close()  # 关
print(crypto) #返回list

f=open('./key_pri')
key_pri = f.readlines()  # 直接将文件中按行读到list里，效果与方法2一样
f.close()  # 关
print(key_pri) #返回list

passwd = rsa1.rsaDecrypt(crypto, key_pri)

#这种硬编码性质的链接并不安全因此在此
conn = pymysql.connect(host='localhost',port=3306, user='root', password=passwd)
# 创建游标
cursor = conn.cursor()
# 创建数据库的sql(使用if判断是否已经存在数据库，数据库不存在时才会创建，否则会报错)
sql = "CREATE DATABASE IF NOT EXISTS pokemon"  # pokemon是要创建的数据库名字

# 执行创建数据库的sql语句
cursor.execute(sql)
conn.commit()       #进行提交
cursor.close()    #关闭游标
conn.close()   #关闭数据库连接
