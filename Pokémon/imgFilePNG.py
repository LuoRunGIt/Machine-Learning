# 处理PNG图片
# 将PNG转换为JPG

import os
from PIL import Image


# 获取指定目录下的所有png图片
def get_all_png_files(dir):
    files_list = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if os.path.splitext(file)[1] == '.png':
                files_list.append(os.path.join(root, file))
    return files_list


# 批量转换png图片为jpg格式
def png2jpg(files_list):
    for file in files_list:
        img = Image.open(file)

        new_file = os.path.splitext(file)[0] + '.jpg'
        img.convert('RGB').save(new_file)

def delPNG(files_list):
    for file in files_list:
        os.remove( file)

file_list=get_all_png_files("E:\\BaiduNetdiskDownload\\baokemeng\\pokemon")
#print(file_list)

png2jpg(file_list)

delPNG(file_list)