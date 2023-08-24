import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses
from tensorflow.keras.callbacks import EarlyStopping

def preprocess(img):
    img = tf.io.read_file(img)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [244, 244])

    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_crop(img, [224,224,3])

    img = tf.cast(img, dtype=tf.float32) / 255.

    return img


img = '3.jpg'
x = preprocess(img)
x = tf.reshape(x, [1, 224, 224, 3])
network = tf.keras.models.load_model('model.h5')
logits = newnet.predict(x)
prob = tf.nn.softmax(logits, axis=1)
print(prob)
max_prob_index = np.argmax(prob, axis=-1)[0]
prob = prob.numpy()
max_prob = prob[0][max_prob_index]
print(max_prob)
max_index = np.argmax(logits, axis=-1)[0]
name = ['妙蛙种子', '小火龙', '超梦', '皮卡丘', '杰尼龟']
print(name[max_index])


