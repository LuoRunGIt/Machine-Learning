import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import librosa.feature
# 特征中增加差分量
y,fs = librosa.load('gu.wav',sr=16000)
win_length =512
hop_length = 160
n_fft = 512
n_mels = 128
n_mfcc = 20
mfcc = librosa.feature.mfcc(y=y,
                             sr=fs,
                             n_mfcc=n_mfcc,
                             win_length = win_length,
                             hop_length =hop_length,
                             n_fft = n_fft,
                             n_mels = n_mels,
                             dct_type=1
                             )
# 一阶差分
mfcc_deta =  librosa.feature.delta(mfcc)
# 二阶差分
mfcc_deta2 = librosa.feature.delta(mfcc,order=2)
#
# 特征拼接
mfcc_d1_d2 = np.concatenate([mfcc,mfcc_deta,mfcc_deta2],axis=0)
fig = plt.figure()
img = librosa.display.specshow(mfcc_d1_d2, x_axis='time',hop_length=hop_length, sr=fs)
fig.colorbar(img)
plt.show()