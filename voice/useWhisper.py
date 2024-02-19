import whisper
import os
#model = whisper.load_model("base",device="cpu")
#model = whisper.load_model("base")
model = whisper.load_model("small",device="cpu")
#E:\myprogramme\python_workspace\pythonProject\voice
file_path= "./x.mp3"
#运行需要管理员权限
try:
    # 检查文件是否存在
    if  not os.path.exists(file_path):
        raise FileNotFoundError("文件bu存在")
except FileNotFoundError as e:
    print("文件不存在:", e)

audio = whisper.load_audio("./01.wav")
audio = whisper.pad_or_trim(audio)
result = model.transcribe("./01.wav")
print(result["text"])