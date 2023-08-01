# 词云
# 使⽤jieba中的词性标注功能
import matplotlib.pyplot as plt
import jieba.posseg as pseg
import pandas as pd
# 导⼊jieba⽤于分词
# 导⼊chain⽅法⽤于扁平化列表
import jieba
from itertools import chain

train_data = pd.read_csv("./train.tsv", sep="\t")
valid_data = pd.read_csv("./dev.tsv", sep="\t")
# 进⾏训练集的句⼦进⾏分词, 并统计出不同词汇的总数
train_vocab = set(chain(*map(lambda x: jieba.lcut(x),train_data["sentence"])))
print("训练集共包含不同词汇总数为：", len(train_vocab))
# 进⾏验证集的句⼦进⾏分词, 并统计出不同词汇的总数
valid_vocab = set(chain(*map(lambda x: jieba.lcut(x),valid_data["sentence"])))
print("训练集共包含不同词汇总数为：", len(valid_vocab))
def get_a_list(text):
    """⽤于获取形容词列表"""
    # 使⽤jieba的词性标注⽅法切分⽂本,获得具有词性属性flag和词汇属性word的对象,
    # 从⽽判断flag是否为形容词,来返回对应的词汇
    r = []
    for g in pseg.lcut(text):
        if g.flag == "a":
            r.append(g.word)
    return r


# 导⼊绘制词云的⼯具包
from wordcloud import WordCloud


def get_word_cloud(keywords_list):
    # 实例化绘制词云的类, 其中参数font_path是字体路径, 为了能够显示中⽂,
    # max_words指词云图像最多显示多少个词, background_color为背景颜⾊
    wordcloud = WordCloud(font_path="./SimHei.ttf", max_words=100, background_color="white")  # 将传⼊的列表转化成词云⽣成器需要的字符串形式
    keywords_string = " ".join(keywords_list)
    # ⽣成词云
    wordcloud.generate(keywords_string)
    # 绘制图像并显示
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


# 获得训练集上正样本
p_train_data = train_data[train_data["label"] == 1]["sentence"]
# 对正样本的每个句⼦的形容词
train_p_a_vocab = chain(*map(lambda x: get_a_list(x), p_train_data))
# print(train_p_n_vocab)
# 获得训练集上负样本
n_train_data = train_data[train_data["label"] == 0]["sentence"]
# 获取负样本的每个句⼦的形容词
train_n_a_vocab = chain(*map(lambda x: get_a_list(x), n_train_data))
# 调⽤绘制词云函数
get_word_cloud(train_p_a_vocab)
get_word_cloud(train_n_a_vocab)
