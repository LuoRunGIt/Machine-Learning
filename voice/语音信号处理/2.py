import librosa
#librosa是一个音频信号处理库
import soundfile as sf
import sklearn
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

out_path = f"xxx1.mp3"
input_audio = f'../x.mp3'
# 加载音频文件
audio, sr = librosa.load(input_audio)

# 提取梅尔频谱图
mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)

# 可视化梅尔频谱图
librosa.display.specshow(librosa.power_to_db(mel_spec, ref=np.max), y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.show()

