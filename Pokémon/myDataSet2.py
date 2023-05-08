import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

random_data = np.random.randn(10, 3)
print(random_data)
print("#" * len(random_data))


class MyDataSet(Dataset):
    def __init__(self, loaded_data):
        self.data = loaded_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


custom_dataset = MyDataSet(random_data)

for i in range(len(custom_dataset)):
    print(custom_dataset[i])

train_size = int(len(custom_dataset) * 0.5)
validate_size = int(len(custom_dataset) * 0.2)
test_size = len(custom_dataset) - validate_size - train_size
train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(custom_dataset,
                                                                              [train_size, validate_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
validate_loader = DataLoader(validate_dataset, batch_size=1, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

print(len(train_loader))
print(len(validate_loader))
print(len(test_loader))

for i, train_sample in enumerate(train_loader):
    print("i:{} {}".format(i, train_sample))
# enumerate 可以遍历出数据和数据下标
for j, validate_sample in enumerate(validate_loader):
    print("j:{} {}".format(j, validate_sample))

for k, test_sample in enumerate(test_loader):
    print("k:{} {}".format(k, test_sample))
