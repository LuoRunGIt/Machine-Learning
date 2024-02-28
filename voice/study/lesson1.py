import os
import torchaudio
import IPython.display as ipd
import matplotlib.pyplot as plt

'''PyTorch 提供了各种可用的示例数据集，这在您尝试学习和使用不同的音频模型时非常有用。
我们将使用语音命令示例数据集，并将完整数据集下载到本地目录中。
请注意，我们仅使用 yes 和 no 类来创建二元分类模型。'''


# Create a data folder
def create_folder():
    default_dir = os.getcwd()
    folder = 'data'
    print(f'Data directory will be: {default_dir}/{folder}')

    if os.path.isdir(folder):
        print("Data folder exists.")
    else:
        print("Creating folder.")
        os.mkdir(folder)


# create_folder()
default_dir = os.getcwd()
folder = 'data'
# 下载数据集
trainset_speechcommands = torchaudio.datasets.SPEECHCOMMANDS(f'./{folder}/', download=True)

# 可视化数据集中可用的类
os.chdir(f'./{folder}/SpeechCommands/speech_commands_v0.02/')
labels = [name for name in os.listdir('.') if os.path.isdir(name)]
# back to default directory
os.chdir(default_dir)
print(f'Total Labels: {len(labels)} \n')
print(f'Label Names: {labels}')

'''
您以前可能使用过波形文件。这是一种格式，我们保存模拟音频的数字表示以供共享和播放。我们将在本教程中使用的语音命令数据集存储在波形文件中，这些文件都为一秒或更短。
让我们加载其中一个波形文件，看看波形的张量是什么样子的。我们使用 torchaudio.load 加载文件，它将音频文件加载到中。
torch.Tensor 对象。TorchAudio 负责实施，因此您无需担心。torch.load 函数以张量和sample_rate的 int 形式返回波形。
'''
filename = "./data/SpeechCommands/speech_commands_v0.02/yes/00f0204f_nohash_0.wav"

'''
要加载音频数据，您可以使用torchaudio.load().
该函数接受类似路径的对象或类似文件的对象作为输入。
返回值是波形 ( Tensor) 和采样率 ( int) 的元组。
默认情况下，生成的张量对象具有dtype=torch.float32，其值范围为。[-1.0, 1.0]
'''
# num_frames 表示读取的帧数，单位fps

waveform, sample_rate = torchaudio.load(filepath=filename, num_frames=3)
print(f'waveform tensor with 3 frames:  {waveform} \n')
waveform, sample_rate = torchaudio.load(filepath=filename, num_frames=3, frame_offset=2)
print(f'waveform tensor with 2 frame_offsets: {waveform} \n')
waveform, sample_rate = torchaudio.load(filepath=filename)
print(f'waveform tensor:  {waveform}')


# 注意 如果出现UserWarning: No audio backend is available. 则说明需要补充安装额外的输入插件 PySoundFile

# Plot the waveform 绘画波形
def plot_audio(filename):
    waveform, sample_rate = torchaudio.load(filename)

    print("Shape of waveform: {}".format(waveform.size()))
    print("Sample rate of waveform: {}".format(sample_rate))

    plt.figure()
    plt.plot(waveform.t().numpy())
    plt.show()
    return waveform, sample_rate


filename = "./data/SpeechCommands/speech_commands_v0.02/yes/00f0204f_nohash_0.wav"
waveform, sample_rate = plot_audio(filename)
# 这是yes的波形

# jupyter notebook 下可以播放音频
ipd.Audio(waveform.numpy(), rate=sample_rate)
