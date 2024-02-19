import speech_recognition as sr

# 创建Recognizer对象
r = sr.Recognizer()

# 打开麦克风获取输入的语音
with sr.Microphone() as source:
    print("正在录制...")


    # 等待说话结束
    audio = r.listen(source)

    try:
        # 使用Google Web Speech API进行语音识别
        text = r.recognize_google(audio, language='zh-CN')

        # 输出识别结果
        print('识别结果：', text)
    except sr.UnknownValueError:
        print("无法理解语音内容")
    except sr.RequestError as e:
        print("服务器错误：", str(e))