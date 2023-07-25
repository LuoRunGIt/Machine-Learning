import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img=img/2+0.5
    npimg=img.numpy()
    #transpose为维度转置
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()