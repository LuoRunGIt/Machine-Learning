import string
import unicodedata

# 获取所有常⽤字符包括字⺟和常⽤标点
all_letters = string.ascii_letters + ".,;'"
# 获取常⽤字符数量
n_letters = len(all_letters)
# 57
print("n_letters:", n_letters)




def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


s = "Ślusàrski"
a = unicodeToAscii(s)
print(a)