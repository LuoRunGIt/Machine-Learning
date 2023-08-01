def create_ngram_set(input_list):
 """
 description: 从数值列表中提取所有的n-gram特征
 :param input_list: 输⼊的数值列表, 可以看作是词汇映射后的列表,
 ⾥⾯每个数字的取值范围为[1, 25000]
 :return: n-gram特征组成的集合
 eg:
 >>> create_ngram_set([1, 4, 9, 4, 1, 4])
 {(4, 9), (4, 1), (1, 4), (9, 4)}
 """
 return set(zip(*[input_list[i:] for i in range(ngram_range)]))