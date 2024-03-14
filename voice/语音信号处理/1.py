from pydub import AudioSegment

# 导入音频文件
path="../x.mp3"
sound1 = AudioSegment.from_file(path, format="mp3")

# 变成女声，参数pitch可以调节音高，speed可以调节语速，默认都是1
# 可以自己调节
# 给 sound1 增益 3.5 dB
louder_via_method = sound1.apply_gain(+3.5)
#70会导致效果很奇怪，有种杂音，或者说杂音也被放大了70
louder_via_operator = sound1 + 7
#音频时长
k=louder_via_operator.duration_seconds
print(k)
# 保存为新的文件
louder_via_operator.export("gujianbao.mp3", format="mp3")
