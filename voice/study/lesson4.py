# 训练完一个epoch，我们可能会生成模型来进行测试。在测试之前，
# 需要加上model.eval()，否则的话，即使不训练，模型的权值也会改变。这是因为模型中有Batch Normalization层和Dropout层。
'''
model.eval()
test_loss, correct = 0, 0
class_map = ['no', 'yes']

with torch.no_grad():
    for batch, (X, Y) in enumerate(test_dataloader):
        X, Y = X.to(device), Y.to(device)
        pred = model(X)
        print("Predicted:\nvalue={}, class_name= {}\n".format(pred[0].argmax(0),class_map[pred[0].argmax(0)]))
        print("Actual:\nvalue={}, class_name= {}\n".format(Y[0],class_map[Y[0]]))
        break
'''

# 读取录音文件
# 录音文件转换为指定的图片
# 输入模型进行预测
import torch
from torchvision import datasets, models, transforms

from cnn import CNNet
import torch.nn as nn
import torch.nn.functional as F
'''
class CNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(51136, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)

'''
def main():
    # 将声谱图图像加载到数据加载器中进行训练
    data_path = './data/spectrograms'  # looking in subfolder train

    yes_no_dataset = datasets.ImageFolder(
        root=data_path,
        transform=transforms.Compose([transforms.Resize((201, 81)),
                                      transforms.ToTensor()
                                      ])
    )
    print(yes_no_dataset)

    # 基于每个音频类的文件夹自动创建图像类标签和索引。我们将使用class_to_idx来查看图像数据集的类映射。
    class_map = yes_no_dataset.class_to_idx

    print("\nClass category and index of the images: {}\n".format(class_map))

    # 划分训练集和测试集
    # split data to test and train
    # use 80% to train
    train_size = int(0.8 * len(yes_no_dataset))
    test_size = len(yes_no_dataset) - train_size
    yes_no_train_dataset, yes_no_test_dataset = torch.utils.data.random_split(yes_no_dataset, [train_size, test_size])

    print("Training size:", len(yes_no_train_dataset))
    print("Testing size:", len(yes_no_test_dataset))

    test_dataloader = torch.utils.data.DataLoader(
        yes_no_test_dataset,
        batch_size=15,
        num_workers=2,
        shuffle=True
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #cnn = CNNet().to(device)
    PATH = './models/simple_model.pth'

    cnn = torch.load(PATH)
    cnn.eval()

    test_loss, correct = 0, 0

    class_map = ['no', 'yes']

    i=0
    for batch, (X, Y) in enumerate(test_dataloader):
            X, Y = X.to(device), Y.to(device)
            pred = cnn(X)
            print("Predicted:\nvalue={}, class_name= {}\n".format(pred[0].argmax(0), class_map[pred[0].argmax(0)]))
            print("Actual:\nvalue={}, class_name= {}\n".format(Y[0], class_map[Y[0]]))
            i=i+1
    print("测试集数量",i)

if __name__=="__main__":
    main()