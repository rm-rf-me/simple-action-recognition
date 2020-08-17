# -*- coding: utf-8 -*-
# @Auther   : liou

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

dtype = torch.FloatTensor

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

if __name__ == '__main__':
    x = torch.randn(32, 3, 64, 64).type(dtype)
    x_var = Variable(x.type(dtype))
    ans = fixed_model(x_var)

    print(np.array(ans.size()))
    np.array_equal(np.array(ans.size()), np.array([32, 10]))