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


class ActionClipDataset(Dataset):

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

    def __len__(self):  # 同样的重载__len__方法。
        return self.length  # 此时的长度就不再是之前的三倍了，对每一个clip我们都要用到三章图像。

    def __getitem__(self, idx):  # 同样的重载__getitem__方法。

        folder = idx + 1
        folder = format(folder, '05d')
        clip = []
        if len(self.labels) != 0:
            Label = self.labels[idx][0] - 1
        for i in range(3):  # 循环提取三张图像。
            imidx = i + 1
            imgname = str(imidx) + '.jpg'
            img_path = os.path.join(self.root_dir,
                                    folder, imgname)
            image = Image.open(img_path)
            image = np.array(image)
            clip.append(image)
        if self.transform:
            clip = np.asarray(clip)
            clip = np.transpose(clip, (0, 3, 1, 2))
            clip = torch.from_numpy(np.asarray(clip))
        if len(self.labels) != 0:
            sample = {'clip': clip, 'Label': Label, 'folder': folder}
        else:
            sample = {'clip': clip, 'folder': folder}
        return sample


clip_dataset = ActionClipDataset(root_dir='./data/trainClips/', \
                                 labels=label_train, transform=T.ToTensor())

clip_dataloader = DataLoader(clip_dataset, batch_size=4,
                        shuffle=True, num_workers=4)
clip_dataset_train=ActionClipDataset(root_dir='./data/trainClips/',labels=label_train,transform=T.ToTensor())

clip_dataloader_train = DataLoader(clip_dataset_train, batch_size=16,
                        shuffle=True, num_workers=4)
clip_dataset_val=ActionClipDataset(root_dir='./data/valClips/',labels=label_val,transform=T.ToTensor())

clip_dataloader_val = DataLoader(clip_dataset_val, batch_size=16,
                        shuffle=True, num_workers=4)
clip_dataset_test=ActionClipDataset(root_dir='./data/testClips/',labels=[],transform=T.ToTensor())

clip_dataloader_test = DataLoader(clip_dataset_test, batch_size=16,
                        shuffle=False, num_workers=4)

dtype = torch.cuda.FloatTensor # 这是pytorch所支持的cpu数据类型中的浮点数类型。

print_every = 100   # 这个参数用于控制loss的打印频率，因为我们需要在训练过程中不断的对loss进行检测。

def reset(m):   # 这是模型参数的初始化
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

class Flatten3d(nn.Module):
    def forward(self, x):
        N, C, D, H, W = x.size()
        return x.view(N, -1)

fixed_model_3d = nn.Sequential(
    nn.Conv3d(in_channels = 3, out_channels = 50, kernel_size = 2, stride = 1),
    nn.ReLU(inplace=True),
    nn.MaxPool3d((1, 2, 2), stride = 2),
    nn.Conv3d(in_channels = 50, out_channels = 100, kernel_size = (1, 3, 3), stride = 1),
    nn.ReLU(inplace = True),
    nn.MaxPool3d((1, 3, 3), stride = 2),
    nn.Dropout3d(0.1),
    Flatten3d(),
    nn.ReLU(inplace=True),
    nn.Linear(19600, 10),
    nn.LogSoftmax()
)

fixed_model_3d = fixed_model_3d.type(dtype)


def train_3d(model, loss_fn, optimizer, dataloader, num_epochs=1):
    for epoch in range(num_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        check_accuracy_3d(fixed_model_3d, clip_dataloader_val)
        model.train()
        for t, sample in enumerate(dataloader):
            x_var = Variable(sample['clip'].type(dtype))
            y_var = Variable(sample['Label'].type(dtype).long())

            scores = model(x_var)

            loss = loss_fn(scores, y_var)
            if (t + 1) % print_every == 0:
                print('t = %d, loss = %.4f' % (t + 1, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def check_accuracy_3d(model, loader):
    num_correct = 0
    num_samples = 0
    model.eval()
    for t, sample in enumerate(loader):
        x_var = Variable(sample['clip'].type(dtype))
        y_var = sample['Label'].type(dtype)
        y_var = y_var.cpu()
        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)

        num_correct += (preds.numpy() == y_var.numpy()).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

optimizer = optim.Adam(fixed_model_3d.parameters(), lr=1e-4)

loss_fn = nn.CrossEntropyLoss()


torch.cuda.random.manual_seed(782374)
fixed_model_3d.apply(reset)
fixed_model_3d.train()
train_3d(fixed_model_3d, loss_fn, optimizer,clip_dataloader_train, num_epochs=7)
fixed_model_3d.eval()
check_accuracy_3d(fixed_model_3d, clip_dataloader_val)
