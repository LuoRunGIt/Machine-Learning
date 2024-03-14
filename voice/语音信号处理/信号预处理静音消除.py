import librosa
import librosa.display
import matplotlib.pyplot as plt

# 静音消除（前后）
y, fs = librosa.load('./gujianbao.mp3', sr=None
                     , mono=False)
print(len(y))  # 这个是整体语音的长度
# 当db值小于阈值则认为是静音
yt, index = librosa.effects.trim(y, top_db=30)
print(index)  # 这里返回的是现有语音的范围[26624 61952]
fig, axs = plt.subplots(nrows=2, ncols=1)
librosa.display.waveshow(y, sr=fs, ax=axs[0], color='blue')
librosa.display.waveshow(yt, sr=fs, ax=axs[1], color='red')
plt.show()
