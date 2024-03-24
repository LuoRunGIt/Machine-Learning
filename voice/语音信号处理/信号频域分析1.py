import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

y, fs = librosa.load('./gujianbao.mp3', sr=16000)
frame_t = 25  # 25ms帧长
hop_length_t = 10  # 10ms步近
win_length = int(frame_t * fs / 1000)
hop_length = int(hop_length_t * fs / 1000)
n_fft = int(2 ** np.ceil(np.log2(win_length)))
# 短时傅里叶变换后得到频谱特征
S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length))

fig=plt.figure()
plt.imshow(S,origin='lower',cmap='hot')
plt.show()
#FFT 的结果大都显示在低频部分所以不显眼

#改成对数
S=librosa.amplitude_to_db(S,ref=np.max)
D,N=S.shape
rangeD=np.arange(0,D,20)#20个间隔
rangeN=np.arange(0,N,20)
range_t=rangeN*(hop_length/fs)
range_f=rangeD*(fs/n_fft/1000)

fig1=plt.figure()
plt.xticks(rangeN,range_t)
plt.yticks(rangeD,range_f)

print(S.shape)
plt.imshow(S,origin='lower',cmap='hot')
plt.colorbar()
plt.show()

#自己算太麻烦了，直接使用内置的函数
fig=plt.figure()
librosa.display.specshow(S,y_axis='linear',x_axis='time',hop_length=hop_length,sr=fs)
plt.colorbar
plt.show()
