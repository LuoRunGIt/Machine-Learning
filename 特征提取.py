from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer


def dict_demo():
    data = [{'city': '北京', 'temperature': 100},
            {'city': '上海', 'temperature': 60},
            {'city': '深圳', 'temperature': 30}]
    # 1、实例化⼀个转换器类
    transfer = DictVectorizer(sparse=False)
    # 2、调⽤fit_transform
    data = transfer.fit_transform(data)
    print("返回的结果:\n", data)
    # 打印特征名字
    print("特征名字：\n", transfer.get_feature_names_out())
    return None


dict_demo()  # 结果相当于one-hot编码


def text_count_demo():
    """
    对⽂本进⾏特征抽取，countvetorizer
    :return: None
    """
    data = ["life is short,i like like python", "life is too long,i dislike python", "人生苦短，但充满挑战"]
    # 1、实例化⼀个转换器类
    # transfer = CountVectorizer(sparse=False) # 注意,没有sparse这个参数
    transfer = CountVectorizer()
    # 2、调⽤fit_transform
    data = transfer.fit_transform(data)
    print("⽂本特征抽取的结果：\n", data.toarray())
    print("返回特征名字：\n", transfer.get_feature_names_out())
    return None


text_count_demo()
