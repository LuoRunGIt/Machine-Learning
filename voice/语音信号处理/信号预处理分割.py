import librosa
import librosa.display
import matplotlib.pyplot as plt

# 静音消除（前后）
y, fs = librosa.load('./gujianbao.mp3', sr=None
                     , mono=False)
print(len(y))  # 这个是整体语音的长度

#分割，按20db进行分段
intervals=librosa.effects.split(y,top_db=20)
print(intervals)
#
y_remix=librosa.effects.remix(y,intervals)
fig, axs = plt.subplots(nrows=2, ncols=1)
librosa.display.waveshow(y, sr=fs, ax=axs[0], color='blue')
librosa.display.waveshow(y_remix, sr=fs, ax=axs[1], color='red',offset=intervals[0][0]/fs)

for interval in intervals:
    axs[0].vlines(interval[0]/fs,-1,1,colors='r')
    axs[1].vlines(interval[1]/fs,-1,1,colors='blue')

plt.show()