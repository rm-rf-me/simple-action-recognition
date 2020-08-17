# -*- coding: utf-8 -*-
# @Auther   : liou

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

dtype = torch.FloatTensor

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

if __name__ == '__main__':
    fixed_model_3d = fixed_model_3d.type(dtype)
    x = torch.randn(32,3, 3, 64, 64).type(dtype)
    x_var = Variable(x).type(dtype)
    ans = fixed_model_3d(x_var)
    np.array_equal(np.array(ans.size()), np.array([32, 10]))