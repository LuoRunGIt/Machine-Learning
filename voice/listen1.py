import speech_recognition as sr

# 创建一个Recognizer对象，用于语音识别
r = sr.Recognizer()

# 设置相关阈值
r.non_speaking_duration = 0.3
r.pause_threshold = 0.5

# 创建一个Microphone对象，设置采样率为16000
# 构造函数所需参数device_index=None, sample_rate=None, chunk_size=1024
msr = sr.Microphone(sample_rate=16000)

# 打开麦克风
with msr as source:
    # 如果想连续录音，该段代码使用for循环
    # 进行环境噪音适应，duration为适应时间，不能小于0.5
    # 如果无噪声适应要求，该段代码可以注释
    r.adjust_for_ambient_noise(source, duration=0.5)

    print("开始录音")

    # 使用Recognizer监听麦克风录音
    # phrase_time_limit=None表示不设置时间限制
    audio = r.listen(source, phrase_time_limit=2)#单位为s

    print("录音结束")

    # 将录音数据写入.wav格式文件
    with open("microphone-results.wav", "wb") as f:
        # audio.get_wav_data()获得wav格式的音频二进制数据
        f.write(audio.get_wav_data())

        # 将录音数据写入.raw格式文件
    with open("microphone-results.raw", "wb") as f:
        f.write(audio.get_raw_data())

        # 将录音数据写入.aiff格式文件
    with open("microphone-results.aiff", "wb") as f:
        f.write(audio.get_aiff_data())

        # 将录音数据写入.flac格式文件
    with open("microphone-results.flac", "wb") as f:
        f.write(audio.get_flac_data())