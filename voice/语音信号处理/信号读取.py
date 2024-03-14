import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf

# 双通道，sr就是采样率
y, fs = librosa.load('./gujianbao.mp3', sr=None
                     ,mono=False)
print(y.shape)
print(fs)#采样率 sr为none时为48000，sr为16000时为16000