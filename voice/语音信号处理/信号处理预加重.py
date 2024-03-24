import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

y, fs = librosa.load('./gujianbao.mp3', sr=16000)
# 预加重 pre-emphasis

n_fft = 512
win_length = 512
hop_length = 160


y_filt = librosa.effects.preemphasis(y)
#sf.write("test_trim_pre.wav", y_filt, fs)
S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
S = librosa.amplitude_to_db(np.abs(S))

S_preemp = librosa.stft(y_filt, n_fft=512, hop_length=hop_length, win_length=win_length)
S_preemp = librosa.amplitude_to_db(np.abs(S_preemp))

fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)

librosa.display.specshow(S, sr=fs, hop_length=hop_length, y_axis='linear', x_axis='time', ax=axs[0])
axs[0].set(title='Original signal')

img = librosa.display.specshow(S_preemp, sr=fs, hop_length=hop_length, y_axis='linear', x_axis='time', ax=axs[1])
axs[1].set(title='pre-emphasis signal')
fig.colorbar(img, ax=axs, format="%+2.f dB")
plt.show()