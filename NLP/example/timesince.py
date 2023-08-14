import time
import math
def timeSince(since):
 "获得每次打印的训练耗时, since是训练开始时间"
 # 获得当前时间
 now = time.time()
 # 获得时间差，就是训练耗时
 s = now - since
 # 将秒转化为分钟, 并取整
 m = math.floor(s / 60)
 # 计算剩下不够凑成1分钟的秒数
 s -= m * 60
 # 返回指定格式的耗时
 return '%dm %ds' % (m, s)
since = time.time() - 10*60
period = timeSince(since)
print(period)