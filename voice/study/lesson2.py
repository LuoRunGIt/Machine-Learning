import os
import torch
import torchaudio
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

'''首先，我们将通过过滤掉 nohash 路径下的 yes 和 no 命令来浏览我们在本地目录中下载的音频文件。
然后，我们将文件加载到 torchaudio 数据对象中。这样可以很容易地提取音频的属性（例如，波形和采样率）。'''


# dataset 数据
def load_audio_files(path: str, label: str):
    dataset = []
    walker = sorted(str(p) for p in Path(path).glob(f'*.wav'))

    for i, file_path in enumerate(walker):
        path, filename = os.path.split(file_path)
        speaker, _ = os.path.splitext(filename)
        speaker_id, utterance_number = speaker.split("_nohash_")
        utterance_number = int(utterance_number)

        # Load audio
        waveform, sample_rate = torchaudio.load(file_path)
        dataset.append([waveform, sample_rate, label, speaker_id, utterance_number])

    return dataset


trainset_speechcommands_yes = load_audio_files('./data/SpeechCommands/speech_commands_v0.02/yes', 'yes')
trainset_speechcommands_no = load_audio_files('./data/SpeechCommands/speech_commands_v0.02/no', 'no')

# 4044个yes语音和3941个no语音
print(f'Length of yes dataset: {len(trainset_speechcommands_yes)}')
print(f'Length of no dataset: {len(trainset_speechcommands_no)}')

# 现在，将数据集加载到数据加载器中，用于 yes 和 no 训练样本集。DataLoader 设置要迭代的批处理数，
# 以便通过网络加载数据集以训练模型。我们将批处理大小设置为 1，因为我们希望在一次迭代中加载整个批处理。

trainloader_yes = torch.utils.data.DataLoader(trainset_speechcommands_yes, batch_size=1,
                                              shuffle=True, num_workers=0)

trainloader_no = torch.utils.data.DataLoader(trainset_speechcommands_no, batch_size=1,
                                             shuffle=True, num_workers=0)

print(f'Length of yes dataset: {len(trainloader_yes)}')
print(f'Length of no dataset: {len(trainloader_no)}')
# 为了了解数据的外观，我们将从每个类中获取波形和采样率，并打印出数据集的样本。
# 波形值waveform是一个具有浮点数据类型的张量。
# 在捕获音频信号的格式中，采样率值为16000。
# 标签值是音频中说出的单词的命令分类，是或否。
# ID是音频文件的唯一标识符。

# 一个trainset_speechcommands_yes[]表示一个音频
print("长度", len(trainset_speechcommands_yes[4043]))
yes_waveform = trainset_speechcommands_yes[0][0]
yes_sample_rate = trainset_speechcommands_yes[0][1]
print(f'Yes Waveform: {yes_waveform}')
print(f'Yes Sample Rate: {yes_sample_rate}')
print(f'Yes Label: {trainset_speechcommands_yes[0][2]}')
print(f'Yes ID: {trainset_speechcommands_yes[0][3]} \n')

no_waveform = trainset_speechcommands_no[0][0]
no_sample_rate = trainset_speechcommands_no[0][1]
print(f'No Waveform: {no_waveform}')
print(f'No Sample Rate: {no_sample_rate}')
print(f'No Label: {trainset_speechcommands_no[0][2]}')
print(f'No ID: {trainset_speechcommands_no[0][3]}')

# 转换和可视化
'''
波形是通过采样率和频率生成的，并直观地表示信号。该信号可以以图形格式表示为波形，该波形是随时间变化的信号表示。
音频可以录制在不同的频道中。例如，立体声录音有两个声道：右声道和左声道。
以下是如何使用重采样变换来减小波形的大小，然后将数据绘制成图形以可视化新的波形形状。
'''


def show_waveform(waveform, sample_rate, label):
    print("Waveform: {}\nSample rate: {}\nLabels: {} \n".format(waveform, sample_rate, label))
    new_sample_rate = sample_rate / 10

    # Resample applies to a single channel, we resample first channel here
    channel = 0
    waveform_transformed = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(
        waveform[channel, :].view(1, -1))

    print("Shape of transformed waveform: {}\nSample rate: {}".format(waveform_transformed.size(), new_sample_rate))

    plt.figure()
    plt.plot(waveform_transformed[0, :].numpy())
    plt.show()


show_waveform(yes_waveform, yes_sample_rate, 'yes')
show_waveform(no_waveform, no_sample_rate, 'no')

# Spectrogram 频谱图
'''
那么，什么是声谱图？声谱图将音频文件的频率映射到时间，它允许您按频率可视化音频数据。它是一种图像格式。这张图片是我们将用于对音频文件进行计算机视觉分类的图片。
您可以以灰度或红-绿-蓝（RGB）颜色格式查看光谱图图像。
每个声谱图图像都有助于以彩色模式显示声音信号产生的不同特征。
卷积神经网络（CNN）将图像中的颜色模式视为用于训练模型以对音频进行分类的特征。
让我们使用PyTorch torchaudio.transforms函数将波形转换为声谱图图像格式。'''


def show_spectrogram(waveform_classA, waveform_classB):
    yes_spectrogram = torchaudio.transforms.Spectrogram()(waveform_classA)
    print("\nShape of yes spectrogram: {}".format(yes_spectrogram.size()))

    no_spectrogram = torchaudio.transforms.Spectrogram()(waveform_classB)
    print("Shape of no spectrogram: {}".format(no_spectrogram.size()))

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("Features of {}".format('no'))
    plt.imshow(yes_spectrogram.log2()[0, :, :].numpy(), cmap='viridis')

    plt.subplot(1, 2, 2)
    plt.title("Features of {}".format('yes'))
    plt.imshow(no_spectrogram.log2()[0, :, :].numpy(), cmap='viridis')

    plt.show()


# y轴是音频的频率。
# x轴是音频的时间。
# 图像的强度表示音频的振幅。在下面的频谱图图像中，黄色的高浓度说明了音频的幅度。

#show_spectrogram(yes_waveform, no_waveform)

# Mel spectrogram 梅尔光谱图
'''
梅尔谱图也是一个频率到时间，但频率被转换为梅尔标度。梅尔音阶根据对音阶或旋律声音的感知来改变频率。
这将频率转换为梅尔标度，然后创建频谱图图像。'''


def show_melspectrogram(waveform, sample_rate):
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate)(waveform)
    print("Shape of spectrogram: {}".format(mel_spectrogram.size()))

    plt.figure()
    plt.imshow(mel_spectrogram.log2()[0, :, :].numpy(), cmap='viridis')
    plt.show()


#show_melspectrogram(yes_waveform, yes_sample_rate)

#Mel-frequency cepstral coefficients MFCC 美尔倒谱系数
'''对MFCC的作用的简化解释是，它取我们的频率，应用变换，结果是由频率产生的频谱的振幅。让我们看看这是什么样子。
A simplified explanation of what the MFCC does is that it takes our frequency, applies transforms, 
and the result is the amplitudes of the spectrum created from the frequency. 
Let's take a look at what this looks like.'''


def show_mfcc(waveform, sample_rate):
    mfcc_spectrogram = torchaudio.transforms.MFCC(sample_rate=sample_rate)(waveform)
    print("Shape of spectrogram: {}".format(mfcc_spectrogram.size()))

    plt.figure()
    fig1 = plt.gcf()
    plt.imshow(mfcc_spectrogram.log2()[0, :, :].numpy(), cmap='viridis')

    plt.figure()
    plt.plot(mfcc_spectrogram.log2()[0, :, :].numpy())
    plt.draw()
    plt.show()
#show_mfcc(no_waveform,  no_sample_rate)

#从声谱图创建图像
'''此时，您可以更好地了解音频数据，以及可以对其使用的不同转换。现在，让我们创建用于分类的图像。
以下是创建用于分类的声谱图图像或MFCC图像的两个不同功能。您将使用光谱图图像来训练我们的模型。如果你想的话，可以自己玩MFCC分类。'''


def create_spectrogram_images(trainloader, label_dir):
    # make directory
    directory = f'./data/spectrograms/{label_dir}/'
    if (os.path.isdir(directory)):
        print("Data exists for", label_dir)
    else:
        os.makedirs(directory, mode=0o777, exist_ok=True)

        for i, data in enumerate(trainloader):
            waveform = data[0]
            sample_rate = data[1][0]
            label = data[2]
            ID = data[3]

            # create transformed waveforms
            spectrogram_tensor = torchaudio.transforms.Spectrogram()(waveform)

            fig = plt.figure()
            plt.imsave(f'./data/spectrograms/{label_dir}/spec_img{i}.png',
                       spectrogram_tensor[0].log2()[0, :, :].numpy(), cmap='viridis')

#从mfcc创建图像
def create_mfcc_images(trainloader, label_dir):
    # make directory
    os.makedirs(f'./data/mfcc_spectrograms/{label_dir}/', mode=0o777, exist_ok=True)

    for i, data in enumerate(trainloader):
        waveform = data[0]
        sample_rate = data[1][0]
        label = data[2]
        ID = data[3]

        mfcc_spectrogram = torchaudio.transforms.MFCC(sample_rate=sample_rate)(waveform)

        plt.figure()
        fig1 = plt.gcf()
        plt.imshow(mfcc_spectrogram[0].log2()[0, :, :].numpy(), cmap='viridis')
        plt.draw()
        fig1.savefig(f'./data/mfcc_spectrograms/{label_dir}/spec_img{i}.png', dpi=100)
        plt.close(fig1)

        # spectorgram_train.append([spectrogram_tensor, label, sample_rate, ID])

#create_spectrogram_images(trainloader_yes, 'yes')
create_spectrogram_images(trainloader_no, 'no')