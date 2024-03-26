# RNN构建的人名分类器
'''
第⼀步: 导⼊必备的⼯具包.
第⼆步: 对data⽂件中的数据进⾏处理，满⾜训练要求.
第三步: 构建RNN模型(包括传统RNN, LSTM以及GRU).
第四步: 构建训练函数并进⾏训练.
第五步: 构建评估函数并进⾏预测.
'''

from io import open
import glob
import os
import string
import unicodedata
import random
import time
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 获取所有常⽤字符包括字⺟和常⽤标点
all_letters = string.ascii_letters + " .,;'"
# 获取常⽤字符数量
n_letters = len(all_letters)
# 57,26个英文字母的大小写和.,;’空格
# print(all_letters)
# print("n_letters:", n_letters)

# 训练数据的存储地址
data_path = "./data/names/"

'''======================================================步骤1-1 去除部分语言中的重音======================================='''


# 函数作用是去掉语言中的一些重音标记
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def readLines(filename):
    '''从指定文件中读取每一行加载到内容中形成列表'''

    # 打开指定⽂件并读取所有内容, 使⽤strip()去除两侧空⽩符, 然后以'\n'进⾏切分
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    # 对应每⼀个lines列表中的名字进⾏Ascii转换, 使其规范化.最后返回⼀个名字列表
    return [unicodeToAscii(line) for line in lines]


filename = data_path + "Chinese.txt"
lines = readLines(filename)
# print(lines)
# 这里是第一步处理后的产物['Ang', 'AuYong', 'Bai', 'Ban', 'Bao', 'Bei', 'Bian',

'''================================步骤1-2构建人名和所属语言的列表的字典=================================


如：{"English":["Lily","Susan"],"Chinese":["Zhang"，"Wang","Zhao"]}

'''
category_lines = {}

# all_categories形如：["English","Chinese"]
all_categories = []

# 读取指定路径下所有txt文件，使用glob，path中可以使用的正则表达式
for filename in glob.glob(data_path + '*.txt'):
    # 获取每个文件的文件名
    # [0]表示文件名称，[1]为.txt
    category = os.path.splitext(os.path.basename(filename))[0]
    # print(category)
    all_categories.append(category)
    # 读取每个文件的内容，形成名字列表
    # 点评：能这么写的原因主要是所有名字都是按文件名进行了分类了
    lines = readLines(filename)
    # 按照对应的类别，将名字列表写入到category_lines字典中
    category_lines[category] = lines

'''
all_categories 表示18个语种['Arabic', 'Chinese', 'Czech', 'Dutc
n_categories =18
category_lines[category]则为目标字典
'''
n_categories = len(all_categories)
# print(all_categories)
# print("n_categories:", n_categories)
# 随便查看其中的⼀些内容，[5:]从第五个开始 [:5]到第五个为止
# print(category_lines['Italian'][:5])
# print(category_lines)

'''================================步骤1-3人名转onehot编码================================='''


def lineToTensor(line):
    """将⼈名转化为对应onehot张量表示, 参数line是输⼊的⼈名"""
    # ⾸先初始化⼀个0张量, 它的形状(len(line), 1, n_letters)， 总共多少字母(如ab 就是2个字母)，每个字母代表1，有多少字母就有多少n_letters(56)
    # 代表⼈名中的每个字⺟⽤⼀个1 x n_letters的张量表示.
    # 这里1 x n_letters为固定维度用于表示字母
    tensor = torch.zeros(len(line), 1, n_letters)
    # 遍历这个⼈名中的每个字符索引和字符
    for li, letter in enumerate(line):
        # 使⽤字符串⽅法find找到每个字符在all_letters中的索引
        # 它也是我们⽣成onehot张量中1的索引位置
        tensor[li][0][all_letters.find(letter)] = 1
    # 返回结果
    return tensor


line = "Baio"
line_tensor = lineToTensor(line)
print("line_tensot:", line_tensor)
print(line_tensor.shape)  # torch.Size([4, 1, 56])

'''================================步骤2-1模型构建================================='''


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """初始化函数中有4个参数, 分别代表RNN输⼊最后⼀维尺⼨, RNN的隐层最后⼀维尺⼨,
        RNN层数
        input_size: 输⼊张量x中特征维度的⼤⼩.
        hidden_size: 隐层张量h中特征维度的⼤⼩.
        num_layers: 隐含层的数量.
        nonlinearity: 激活函数的选择, 默认是tanh
        """
        super(RNN, self).__init__()
        # 将hidden_size与num_layers传⼊其中
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 实例化预定义的nn.RNN, 它的三个参数分别是input_size, hidden_size,num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        # 实例化nn.Linear, 这个线性层⽤于将nn.RNN的输出维度转化为指定的输出维度
        self.linear = nn.Linear(hidden_size, output_size)
        # 实例化nn中预定的Softmax层, ⽤于从输出层获得类别结果
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        """完成传统RNN中的主要逻辑, 输⼊参数input代表输⼊张量, 它的形状是1 x
           n_letters
            hidden代表RNN的隐层张量, 它的形状是self.num_layers x 1 x
           self.hidden_size"""
        # 因为预定义的nn.RNN要求输⼊维度⼀定是三维张量, 因此在这⾥使⽤unsqueeze(0)扩展⼀个维度
        input = input.unsqueeze(0)
        # 将input和hidden输⼊到传统RNN的实例化对象中，如果num_layers=1, rr恒等于hn
        rr, hn = self.rnn(input, hidden)
        # 将从RNN中获得的结果通过线性变换和softmax返回，同时返回hn作为后续RNN的输⼊
        return self.softmax(self.linear(rr)), hn

    def initHidden(self):
        """初始化隐层张量"""
        # 初始化⼀个（self.num_layers, 1, self.hidden_size）形状的0张量
        return torch.zeros(self.num_layers, 1, self.hidden_size)


'''=========================================步骤2-2 模型实例化======================================='''
# 因为是onehot编码, 输⼊张量最后⼀维的尺⼨就是n_letters
input_size = n_letters
# 定义隐层的最后⼀维尺⼨⼤⼩
n_hidden = 128
# 输出尺⼨为语⾔类别总数n_categories
output_size = n_categories
# num_layer使⽤默认值, num_layers = 1

# 假如我们以⼀个字⺟B作为RNN的⾸次输⼊, 它通过lineToTensor转为张量
# 因为我们的lineToTensor输出是三维张量, ⽽RNN类需要的⼆维张量
# 因此需要使⽤squeeze(0)降低⼀个维度
# 注意这里squeeze只能给单维度降维，即b可以从[1,1,57]降维为[1,57]，但是ab 无法从[2,1,57]降维
# []所以我认为unsqueeze的升维度也是一样的吧
#
input = lineToTensor('b').squeeze(0)
print("b", input, 'input的维度', input.shape)
# 初始化⼀个三维的隐层0张量, 也是初始的细胞状态张量,LSTM才会用到c
hidden = c = torch.zeros(1, 1, n_hidden)

# 注意这里字输入是一个1x57 的张量而是不是 1x1x57的张量，但是rnn的默认输入维度又是3维的所以在forward里又扩展回了3维
# 这里先降维又升维属实是把我看傻了，迷之操作
# 这里输入a,b 一类的单个字母可以运行，因为单个字母可以被降维，但是长一点的字母就无法运行了
rnn = RNN(n_letters, n_hidden, n_categories)
rnn_output, next_hidden = rnn(input, hidden)
print("rnn:", rnn_output)

'''=========================================步骤3-1 训练-数据集构建======================================='''


def categoryFromOutput(output):
    """从输出结果中获得指定类别, 参数为输出张量output"""
    # 从输出张量中返回最⼤的值和索引对象, 我们这⾥主要需要这个索引
    top_n, top_i = output.topk(1)
    # top_i对象中取出索引的值
    category_i = top_i[0].item()
    # 根据索引值获得对应语⾔类别, 返回语⾔类别和索引值
    return all_categories[category_i], category_i


# 随机生成训练数据
def randomTrainingExample():
    """该函数⽤于随机产⽣训练数据"""
    '''输出结果为：类别如Russian、名字如Tuikin、类别的索引terson([14]),名字的张量[6,1,57]'''
    # ⾸先使⽤random的choice⽅法从all_categories随机选择⼀个类别
    category = random.choice(all_categories)
    # 然后在通过category_lines字典取category类别对应的名字列表
    # 之后再从列表中随机取⼀个名字
    line = random.choice(category_lines[category])
    # 接着将这个类别在所有类别列表中的索引封装成tensor, 得到类别张量category_tensor
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    # 最后, 将随机取到的名字通过函数lineToTensor转化为onehot张量表示
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor


# category, line, category_tensor, line_tensor=randomTrainingExample()
# print(category, line, category_tensor, line_tensor.shape)

'''=========================================步骤3-2 训练-训练函数======================================='''

# 构建传统RNN训练函数
# 定义损失函数
# 定义损失函数为nn.NLLLoss，因为RNN的最后⼀层是nn.LogSoftmax, 两者的内部计算逻辑正好能够吻合.
criterion = nn.NLLLoss()

# 设置学习率为0.005
learning_rate = 0.005


# RNN的训练函数
def trainRNN(category_tensor, line_tensor):
    """定义训练函数, 它的两个参数是category_tensor类别的张量表示, 相当于训练数据的标签,
    line_tensor名字的张量表示, 相当于对应训练数据"""
    # 在函数中, ⾸先通过实例化对象rnn初始化隐层张量
    hidden = rnn.initHidden()
    # 然后将模型结构中的梯度归0
    rnn.zero_grad()
    # 下⾯开始进⾏训练, 将训练数据line_tensor的每个字符逐个传⼊rnn之中, 得到最终结果
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    # 因为我们的rnn对象由nn.RNN实例化得到, 最终输出形状是三维张量, 为了满⾜于category_tensor
    # 进⾏对⽐计算损失, 需要减少第⼀个维度, 这⾥使⽤squeeze()⽅法
    loss = criterion(output.squeeze(0), category_tensor)
    # 损失进⾏反向传播
    loss.backward()
    # 显式的更新模型中所有的参数
    for p in rnn.parameters():
        # 将参数的张量表示与参数的梯度乘以学习率的结果相加以此来更新参数
        p.data.add_(-learning_rate, p.grad.data)
        # 返回结果和损失的值
    return output, loss.item()


# 每次训练的耗时
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


'''=========================================步骤3-3 训练-运行训练函数======================================='''
# 设置训练迭代次数
n_iters = 5000
# 设置结果的打印间隔
print_every = 50
# 设置绘制损失曲线上的制图间隔
plot_every = 10


def train(train_type_fn):
    """训练过程的⽇志打印函数, 参数train_type_fn代表选择哪种模型训练函数, 如，这里用了回调函数
   trainRNN"""
    # 每个制图间隔损失保存列表
    all_losses = []
    # 获得训练开始时间戳
    start = time.time()
    # 设置初始间隔损失为0
    current_loss = 0
    # 从1开始进⾏训练迭代, 共n_iters次这里是1000
    for iter in range(1, n_iters + 1):
        # 通过randomTrainingExample函数随机获取⼀组训练数据和对应的类别
        category, line, category_tensor, line_tensor = randomTrainingExample()
        '''输出结果为：类别如Russian、名字如Tuikin、类别的索引terson([14]),名字的张量[6,1,57]'''
        # 将训练数据和对应类别的张量表示传⼊到train函数中
        output, loss = train_type_fn(category_tensor, line_tensor)
        # 计算制图间隔中的总损失
        current_loss += loss
        # 如果迭代数能够整除打印间隔
        if iter % print_every == 0:
            # 取该迭代步上的output通过categoryFromOutput函数获得对应的类别和类别索引
            guess, guess_i = categoryFromOutput(output)
            # 然后和真实的类别category做⽐较, 如果相同则打对号, 否则打叉号.
            correct = '✓' if guess == category else '✗ (%s)' % category
            # 打印迭代步, 迭代步百分⽐, 当前训练耗时, 损失, 该步预测的名字, 以及是否正确
            print('%d %d%% (%s) %.4f %s / %s %s' % (
                iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))
        # 如果迭代数能够整除制图间隔
        if iter % plot_every == 0:
            # 将保存该间隔中的平均损失到all_losses列表中
            all_losses.append(current_loss / plot_every)
            # 间隔损失重置为0
            current_loss = 0
        # 返回对应的总损失列表和训练耗时
    return all_losses, int(time.time() - start)


# 调⽤train函数, 分别进⾏RNN, LSTM, GRU模型的训练
# 并返回各⾃的全部损失, 以及训练耗时⽤于制图
all_losses1, period1 = train(trainRNN)

# 绘制损失对⽐曲线, 训练耗时对⽐柱张图
# 创建画布0
plt.figure(0)
# 绘制损失对⽐曲线
plt.plot(all_losses1, label="RNN")
plt.legend(loc='upper left')

# 创建画布1
plt.figure(1)
x_data = ["RNN"]
y_data = [period1]
# 绘制训练耗时对⽐柱状图
plt.bar(range(len(x_data)), y_data, tick_label=x_data)

'''=======================================步骤4 构建预测函数=========================================================='''


# 步骤5评估函数
# RNN
def evaluateRNN(line_tensor):
    """评估函数, 和训练函数逻辑相同, 参数是line_tensor代表名字的张量表示"""
    # 初始化隐层张量
    hidden = rnn.initHidden()
    # 将评估数据line_tensor的每个字符逐个传⼊rnn之中
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    # 获得输出结果
    return output.squeeze(0)


def predict(input_line, evaluate, n_predictions=3):
    """预测函数, 输⼊参数input_line代表输⼊的名字,
    n_predictions代表需要取最有可能的top个"""
    # ⾸先打印输⼊
    print('\n> %s' % input_line)
    # 以下操作的相关张量不进⾏求梯度
    with torch.no_grad():
        # 使输⼊的名字转换为张量表示, 并使⽤evaluate函数获得预测输出
        output = evaluate(lineToTensor(input_line))
        # 从预测的输出中取前3个最⼤的值及其索引
        topv, topi = output.topk(n_predictions, 1, True)
        # 创建盛装结果的列表
        predictions = []
        # 遍历n_predictions
        for i in range(n_predictions):
            # 从topv中取出的output值
            value = topv[0][i].item()
            # 取出索引并找到对应的类别
            category_index = topi[0][i].item()
            # 打印ouput的值, 和对应的类别
            print('(%.2f) %s' % (value, all_categories[category_index]))
            # 将结果装进predictions中
            predictions.append([value, all_categories[category_index]])


print("-" * 18)
predict('Luo', evaluateRNN)
