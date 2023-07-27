# 导⼊⽤于对象保存与加载的joblib
import joblib
# 加载之前保存的Tokenizer, 实例化⼀个t对象
vocab = {"周杰伦", "陈奕迅", "王⼒宏", "李宗盛", "吴亦凡", "⿅晗"}
tokenizer_path = "./Tokenizer"
t = joblib.load(tokenizer_path)
token = "李宗盛"
# 使⽤t获得token_index
token_index = t.texts_to_sequences([token])[0][0] - 1
# 初始化⼀个zero_list
zero_list = [0]*len(vocab)
# 令zero_List的对应索引为1
zero_list[token_index] = 1
print(token, "的one-hot编码为:", zero_list)