import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import librosa.feature

# fbank 特征
# 滤波器组
y, fs = librosa.load('./gu.wav', sr=16000)
win_length = 512
hop_length = 160
n_fft = 512
n_mels = 40#有深度网络的情况下一般设128
#梅尔滤波器组，40个滤波器组，每个滤波器组257
melfb = librosa.filters.mel(sr=fs, n_fft=n_fft, n_mels=n_mels, htk=True)
print(melfb.shape)
x = np.arange(melfb.shape[1]) * fs / n_fft
fig = plt.figure()
plt.plot(x, melfb.T)
plt.show()
fig = plt.figure()
#这里y= 不能省略
fbank = librosa.feature.melspectrogram(y=y,
                                       sr=fs,
                                       n_fft=n_fft,
                                       win_length=win_length,
                                       hop_length=hop_length,
                                       n_mels=n_mels)
print(fbank.shape)
fbank_db = librosa.power_to_db(fbank, ref=np.max)
img = librosa.display.specshow(fbank_db, x_axis='time', y_axis='mel', sr=fs, fmax=fs / 2, )
fig.colorbar(img, format='%+2.0f dB')
plt.title('Mel-frequency spectrogram')
plt.show()
