#以加密方式链接数据库
import pymysql
import rsa1
import rsa

def connect_db(dbname='pokemon'):
    with open("crypto.txt", "rb") as x:
        crypto = x.read()
        x.close()
    print(crypto)  # 返回list

    with open("key_pri.pem", "rb") as x:
        e = x.read()
        key_pri = rsa.PrivateKey.load_pkcs1(e)
        x.close()
    print(e)

    passwd = rsa1.rsaDecrypt(crypto, key_pri)

    # 这种硬编码性质的链接并不安全因此在此
    conn = pymysql.connect(host='localhost', port=3306, user='root', password=passwd,database=dbname)

    return conn


