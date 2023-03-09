from mxnet import autograd, nd
from mxnet.gluon import data as gdata
from mxnet.gluon import nn
from mxnet import init
from mxnet.gluon import loss as gloss
from mxnet import gluon

# 这里在初始化数据
# 生成的数据遵循这个规则：y[i]=x[i][0]*2-x[i][1]*3.4+4.2+noise
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
# 随机输入
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
# print(features)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
# 概率分布的标准差，对应于分布的宽度，scale越大越矮胖，scale越小，越瘦高
labels += nd.random.normal(scale=0.01, shape=labels.shape)
# print(labels[10])
batch_size = 10
# 将训练数据的特征和标签组合
dataset = gdata.ArrayDataset(features, labels)
# 随机读取小批量
# 一次取十个看一下效果
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
for X, y in data_iter:
    print(X, y)
    break

# X为[n.2] y为1维度
# 初始化一个网络
net = nn.Sequential()
# 在网络里加了一层，定义这一层的输出为1
net.add(nn.Dense(1))
# 始化模型参数，如线性回归模型中的权重和偏差，即w和b
net.initialize(init.Normal(sigma=0.01))
loss = gloss.L2Loss()  # 平方损失又称L2范数损失,这是个类
#由于单层的原因其实没有触发sigmod？，sgd是随机梯度下降
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
num_epochs = 3

for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        with autograd.record():  # 表示自动求梯度
            # 求误差函数下的梯度
            l = loss(net(X), y)  # 带入的是一组输入和输出
            # print("l为：",l)
            # print("net",net(X),y)
        # 反向传播
        print("l:",l)
        l.backward()  # 仔细看看书2.5章
        # print(l.backward())
        # trainer.step 指的是更新参数，，我现在最大的问题在于他怎么知道要更新参数,
        # #调用Trainer实例的step函数来迭代模型参数 trainer.step(batch_size)
        # batch_size = 10，相当于对上面10个梯度进行参数修改
        trainer.step(batch_size)#step下一步
    l = loss(net(features), labels)
    # 这里其实想表达的是3次训练过程中，误差值的变化情况
    print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))
# 获得第0层的数据
dense = net[0]
#dense.weight.grad()
true_w, dense.weight.data()
true_b, dense.bias.data()
print(true_w, true_b)
q=nd.arange(4.0)
q=q.reshape((2,2))
print(q)
#求出来并非整数解
print(net(q))