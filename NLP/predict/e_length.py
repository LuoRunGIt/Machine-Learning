#长度限制
from keras.preprocessing import sequence
# cutlen根据数据分析中句⼦⻓度分布，覆盖90%左右语料的最短⻓度.
# 这⾥假定cutlen为10
cutlen = 10
def padding(x_train):
 """
 description: 对输⼊⽂本张量进⾏⻓度规范
 :param x_train: ⽂本的张量表示, 形如: [[1, 32, 32, 61], [2, 54, 21, 7,
19]]
 :return: 进⾏截断补⻬后的⽂本张量表示。截断为从后往前截断，不足则补0
 """
 # 使⽤sequence.pad_sequences即可完成
 return sequence.pad_sequences(x_train, cutlen)

# 假定x_train⾥⾯有两条⽂本, ⼀条⻓度⼤于10, ⼀天⼩于10
x_train = [[1, 23, 5, 32, 55, 63, 2, 21, 78, 32, 23, 1],
 [2, 32, 1, 23, 1]]
res = padding(x_train)
print(res)