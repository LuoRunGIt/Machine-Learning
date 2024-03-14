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
