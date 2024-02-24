# RSA
import rsa


def rsaEncrypt(message):
    '''
    RSA加密函数
    传入需要加密的内容，进行RSA加密，并返回密文 & 私钥 & 公钥
    :param message: 需要加密的内容，明文
    :return: 密文 & 私钥 & 公钥
    '''
    key_pub, key_pri = rsa.newkeys(1024)
    pri = key_pri.save_pkcs1()  # 注意这里的格式需要进行转换
    with open("key_pri.pem", "wb") as x:  # 保存私钥
        x.write(pri)
    # print(key_pri.pem)
    # print(key_pub)
    content = message.encode('utf-8')
    crypto = rsa.encrypt(content, key_pub)
    return (crypto, key_pri, key_pub)


def rsaDecrypt(message, key_pri):
    '''
    RSA 解密函数，传入密文 & 私钥，得到明文；
    :param message: 密文
    :param key_pri: 私钥
    :return: 明文
    '''

    content = rsa.decrypt(message, key_pri)
    return content.decode('utf-8')


# 公钥加密，私钥解密
def test(str1=""):
    message = str1

    print('加密前：', message)  # 加密前： I Love China. 我爱你中国！

    crypto, key_pri, key_pub = rsaEncrypt(message)

    print('加密后:',
          crypto)  # 加密后：
    print('秘钥为:',
          key_pri)  # 秘钥为：
    print('公钥为:',
          key_pub)  # 公钥为：

    with open('./crypto.txt', 'wb') as f:
        f.write(crypto)
        f.close()

    content = rsaDecrypt(crypto, key_pri)
    print('明文为：', content)  # 明文为： I Love China. 我爱你中国！

# test("luorun315")
