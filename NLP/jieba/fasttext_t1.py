import fasttext
fasttext.FastText.eprint = lambda x: None
model = fasttext.load_model('fastmodel.bin')
# 模型质量检测
a=model.get_nearest_neighbors('sports')
print(a)
model.get_word_vector("the")
