import fasttext
model=fasttext.train_unsupervised('fil9')
model.save_model('fastmodel.bin')