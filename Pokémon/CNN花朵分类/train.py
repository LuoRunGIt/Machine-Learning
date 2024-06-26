import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
import imageio
import time
import warnings
import random
import sys
import copy
import json
from PIL import Image
# https://www.bilibili.com/video/BV1wo4y187HR/?p=5&spm_id_from=pageDriver&vd_source=624b74f54fd285fe836718dd760b59fd
# 主要是学习利用torch 中data的api实现使用路径划分数据集

device=torch.device("cuda"if torch.cuda.is_available() else "cpu")
print(device)