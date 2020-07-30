# -*- coding: utf-8 -*-
# @Auther   : liou
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader,sampler,Dataset
import torchvision.datasets as dset
import torchvision.transforms as T
import timeit
from PIL import Image
import os
import numpy as np
import scipy.io
import torchvision.models.inception as inception

label_mat=scipy.io.loadmat('./data/q3_2_data.mat')
label_train=label_mat['trLb']
print('train len：',len(label_train))
label_val=label_mat['valLb']
print('val len: ',len(label_val))


class ActionDataset(Dataset):
    """Action dataset."""

    def __init__(self, root_dir, labels=[], transform=None):
        """
        Args:
            root_dir (string): 整个数据的路径。
            labels(list): 图片的标签。
            transform (callable, optional): 想要对数据进行的处理函数。
        """
        self.root_dir = root_dir
        self.transform = transform
        self.length = len(os.listdir(self.root_dir))
        self.labels = labels

    def __len__(self):  # 该方法只需返回数据的数量。
        return self.length * 3  # 因为每个视频片段都包含3帧。

    def __getitem__(self, idx):  # 该方法需要返回一个数据。

        folder = idx // 3 + 1
        imidx = idx % 3 + 1
        folder = format(folder, '05d')
        imgname = str(imidx) + '.jpg'
        img_path = os.path.join(self.root_dir, folder, imgname)
        image = Image.open(img_path)

        if len(self.labels) != 0:
            Label = self.labels[idx // 3][0] - 1
        if self.transform:  # 如果要先对数据进行预处理，则经过transform函数。
            image = self.transform(image)
        if len(self.labels) != 0:
            sample = {'image': image, 'img_path': img_path, 'Label': Label}
        else:
            sample = {'image': image, 'img_path': img_path}
        return sample


image_dataset=ActionDataset(root_dir='./data/trainClips/', labels=label_train,transform=T.ToTensor())
# torchvision.transforms中定义了非常多对图像的预处理方法，这里使用的ToTensor方法为将0～255的RGB值映射到0～1的Tensor类型。


image_dataloader = DataLoader(image_dataset, batch_size=4,
                        shuffle=True, num_workers=4)


image_dataset_train=ActionDataset(root_dir='./data/trainClips/',labels=label_train,transform=T.ToTensor())

image_dataloader_train = DataLoader(image_dataset_train, batch_size=32,
                        shuffle=True, num_workers=4)
image_dataset_val=ActionDataset(root_dir='./data/valClips/',labels=label_val,transform=T.ToTensor())

image_dataloader_val = DataLoader(image_dataset_val, batch_size=32,
                        shuffle=False, num_workers=4)
image_dataset_test=ActionDataset(root_dir='./data/testClips/',labels=[],transform=T.ToTensor())

image_dataloader_test = DataLoader(image_dataset_test, batch_size=32,
                        shuffle=False, num_workers=4)

dtype = torch.cuda.FloatTensor # 这是pytorch所支持的cpu数据类型中的浮点数类型。

print_every = 100   # 这个参数用于控制loss的打印频率，因为我们需要在训练过程中不断的对loss进行检测。

def reset(m):   # 这是模型参数的初始化
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # 读取各个维度。
        return x.view(N, -1)  # -1代表除了特殊声明过的以外的全部维度。

fixed_model_base = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride = 2),
                nn.BatchNorm2d(64),
                nn.Dropout2d(0.3),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride = 2),
                nn.BatchNorm2d(64),
                nn.Dropout2d(0.2),
                nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride=2),
                nn.BatchNorm2d(32),
                Flatten(),
                nn.Linear(2048, 512),
                nn.Linear(512, 64),
                nn.Linear(64, 10),
                nn.LogSoftmax()
            )


fixed_model = fixed_model_base.type(dtype)


def train(model, loss_fn, optimizer, dataloader, num_epochs=1):
    for epoch in range(num_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))

        # 在验证集上验证模型效果
        check_accuracy(fixed_model, image_dataloader_val)

        model.train()  # 模型的.train()方法让模型进入训练模式，参数保留梯度，dropout层等部分正常工作。
        for t, sample in enumerate(dataloader):
            x_var = Variable(sample['image']).to("cuda")  # 取得一个batch的图像数据。
            y_var = Variable(sample['Label'].long()).to("cuda")  # 取得对应的标签。

            scores = model(x_var)  # 得到输出。

            loss = loss_fn(scores, y_var).to("cuda")  # 计算loss。
            if (t + 1) % print_every == 0:  # 每隔一段时间打印一次loss。
                print('t = %d, loss = %.4f' % (t + 1, loss.item()))

            # 三步更新参数。
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def check_accuracy(model, loader):
    num_correct = 0
    num_samples = 0

    model.eval()  # 模型的.eval()方法切换进入评测模式，对应的dropout等部分将停止工作。
    for t, sample in enumerate(loader):
        x_var = Variable(sample['image']).to("cuda")
        y_var = sample['Label']

        scores = model(x_var)
        _, preds = scores.data.max(1)  # 找到可能最高的标签作为输出。

        num_correct += (preds.cpu().numpy() == y_var.numpy()).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))


optimizer = torch.optim.RMSprop(fixed_model_base.parameters(), lr = 0.0001)

loss_fn = nn.CrossEntropyLoss()

x = torch.randn(32, 3, 64, 64).type(dtype)
x_var = Variable(x.type(dtype))
ans = fixed_model(x_var)

print(np.array(ans.size()))
np.array_equal(np.array(ans.size()), np.array([32, 10]))

torch.random.manual_seed(54321)
fixed_model.to("cuda")
fixed_model.apply(reset)
fixed_model.train()
train(fixed_model, loss_fn, optimizer,image_dataloader_train, num_epochs=11)
