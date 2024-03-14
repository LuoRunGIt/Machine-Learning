import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf

# if __name__ == "__main__":

# 去除头尾的静音
y, fs = librosa.load('./gujianbao.mp3', sr=16000)
print(len(y))
yt, index = librosa.effects.trim(y, top_db=30)
print(index)
fig, axs = plt.subplots(nrows=2, ncols=1)
# 这里必须指定颜色，否则会报错AttributeError: ‘_process_plot_var_args‘ object has no attribute ‘prop_cycler‘
librosa.display.waveshow(y, sr=fs, ax=axs[0], color="blue")
axs[0].vlines(index[0] / fs, -0.5, 0.5, colors='r')
axs[0].vlines(index[1] / fs, -0.5, 0.5, colors='r')
librosa.display.waveshow(yt, sr=fs, ax=axs[1], color="red")
sf.write('./gu.wav',yt,fs)
plt.show()
