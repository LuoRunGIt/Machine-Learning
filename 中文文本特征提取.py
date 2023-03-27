from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import jieba


# 准备句⼦，利⽤jieba.cut进⾏分词 实例化CountVectorizer
# 将分词结果变成字符串当作fit_transform的输⼊值

def cut_word(text):
    """
    对中⽂进⾏分词
    "我爱北京天安⻔"————>"我 爱 北京 天安⻔"
    :param text:
    :return: text
    """
    # 不出现提示
    jieba.setLogLevel(jieba.logging.INFO)
    # ⽤结巴对中⽂字符串进⾏分词
    text = " ".join(list(jieba.cut(text)))

    return text


def text_chinese_count_demo1():
    """
    对中⽂进⾏特征抽取
    :return: None
    """
    data = ["今天很残酷，明天更残酷，后天很美好， 但绝对⼤部分是死在明天晚上，所以每个⼈不要放弃今天。",
            "我们看到的从很远星系来的光是在⼏百万年之前发出的， 这样当我们看到宇宙时，我们是在看它的过去。",
            " 如果只⽤⼀种⽅式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    text_list = []
    for sent in data:
        text_list.append(cut_word(sent))
    print(text_list)
    # 1、实例化⼀个转换器类
    # transfer = CountVectorizer(sparse=False)
    transfer = CountVectorizer(stop_words=["看到", "过去"])
    # 2、调⽤fit_transform
    data = transfer.fit_transform(text_list)
    print("⽂本特征抽取的结果：\n", data.toarray())
    print("返回特征名字：\n", transfer.get_feature_names_out())
    return None


text_chinese_count_demo1()


# TF-idf
def text_chinese_count_demo2():
    data = ["今天很残酷，明天更残酷，后天很美好， 但绝对⼤部分是死在明天晚上，所以每个⼈不要放弃今天。",
            "我们看到的从很远星系来的光是在⼏百万年之前发出的， 这样当我们看到宇宙时，我们是在看它的过去。",
            " 如果只⽤⼀种⽅式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    text_list = []
    for sent in data:
        text_list.append(cut_word(sent))
    print(text_list)
    # 1、实例化⼀个转换器类
    # transfer = CountVectorizer(sparse=False)
    transfer = TfidfVectorizer(stop_words=["看到", "过去"])
    # 2、调⽤fit_transform
    data = transfer.fit_transform(text_list)
    print("⽂本特征抽取的结果：\n", data.toarray())
    print("返回特征名字：\n", transfer.get_feature_names_out())
    return None


text_chinese_count_demo2()
