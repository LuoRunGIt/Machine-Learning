import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np

# mono=False双通道，sr就是采样率
y, fs = librosa.load('./gujianbao.mp3', sr=16000)
print(y.shape)
print(fs)  # 采样率 sr为none时为48000，sr为16000时为16000
# sf 只能保存单通道的数据

fig = plt.figure(1)
yaxis = y
xaxis = np.arange(len(y)) * (1 / fs)
plt.plot(xaxis, yaxis)
plt.xlabel("time")
plt.show()

# 从1s开始截取0.05s长度
y1, fs1 = librosa.load('./gujianbao.mp3', sr=16000, offset=1, duration=0.05)
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(3, 1, 1)
librosa.display.waveshow(y, color="red",sr=fs, ax=ax1)
ax2 = fig1.add_subplot(3, 1, 2)
# 这里offset表示信号从1s时开始
librosa.display.waveshow(y1, color="blue", sr=fs1,ax=ax2, offset=1)

#人为控制显示轴的区域
ax3=fig1.add_subplot(3,1,3)
ax3 .set(xlim=[1,2],ylim=[-1.5,1.5])
librosa.display.waveshow(y,sr=fs,ax=ax3,marker="",color='black')
plt.show()
