import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import librosa.feature
# # MFCC 特征
# # 直接计算
y,fs = librosa.load('./gujianbao.mp3',sr=16000)
win_length =512
hop_length = 160
n_fft = 512
n_mels = 128
n_mfcc = 20
mfcc1 = librosa.feature.mfcc(y=y,
                             sr=fs,
                             n_mfcc=n_mfcc,
                             win_length = win_length,
                             hop_length =hop_length,
                             n_fft = n_fft,
                             n_mels = n_mels)
print(mfcc1.shape)
#
# 也可以提前计算好 log-power fbank特征来进行 mfcc的计算
fbank = librosa.feature.melspectrogram(y=y,
                                       sr = fs,
                                       n_fft = n_fft,
                                       win_length = win_length,
                                       hop_length= hop_length,
                                       n_mels = n_mels)
#
print(fbank.shape)
fbank_db  = librosa.power_to_db(fbank, ref=np.max)
#
mfcc2 = librosa.feature.mfcc(S = fbank_db,n_mfcc=20,sr = fs)
print(mfcc2.shape)
fig = plt.figure()
img = librosa.display.specshow(mfcc1, x_axis='time')
fig.colorbar(img)
plt.title("MFCC")
plt.show()
#
# # # MFCC 不同DCT方式
# y,fs = librosa.load('test1.wav',sr=16000)
# win_length =512
# hop_length = 160
# n_fft = 512
# n_mels = 128
# n_mfcc = 20
# mfcc1 = librosa.feature.mfcc(y,
#                              sr=fs,
#                              n_mfcc=n_mfcc,
#                              win_length = win_length,
#                              hop_length =hop_length,
#                              n_fft = n_fft,
#                              n_mels = n_mels,
#                              dct_type=1
#                              )

# mfcc2 = librosa.feature.mfcc(y,
#                              sr=fs,
#                              n_mfcc=n_mfcc,
#                              win_length = win_length,
#                              hop_length =hop_length,
#                              n_fft = n_fft,
#                              n_mels = n_mels,
#                              dct_type=2
#                              )

# mfcc3 = librosa.feature.mfcc(y,
#                              sr=fs,
#                              n_mfcc=n_mfcc,
#                              win_length = win_length,
#                              hop_length =hop_length,
#                              n_fft = n_fft,
#                              n_mels = n_mels,
#                              dct_type=3
#                              )
# fig, axs = plt.subplots(nrows=3, sharex=True, sharey=True)
# img1 = librosa.display.specshow(mfcc1, x_axis='time',ax = axs[0])
# axs[0].set_title("DCT type 1")
# fig.colorbar(img1,ax=axs[0])

# img2 = librosa.display.specshow(mfcc2, x_axis='time',ax=axs[1])
# axs[1].set_title("DCT type 2")
# fig.colorbar(img2,ax=axs[1])

# img3 = librosa.display.specshow(mfcc3, x_axis='time',ax = axs[2])
# axs[2].set_title("DCT type 3")
# fig.colorbar(img3,ax=axs[2])

# plt.show()