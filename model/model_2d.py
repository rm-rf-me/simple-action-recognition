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
                nn.Conv2d(3, 8, kernel_size=7, stride=1), #3*64*64 -> 8*58*58
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride = 2),    # 8*58*58 -> 8*29*29
                nn.Conv2d(8, 16, kernel_size=7, stride=1), # 8*29*29 -> 16*23*23
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride = 2), # 16*23*23 -> 16*11*11
                Flatten(),
                nn.ReLU(inplace=True),
                nn.Linear(1936, 10)     # 1936 = 16*11*11
            )
# 这里模型base.type()方法是设定模型使用的数据类型，之前设定的cpu的Float类型。
# 如果想要在GPU上训练则需要设定cuda版本的Float类型。
fixed_model = fixed_model_base.type(dtype)

if __name__ == '__main__':
    x = torch.randn(32, 3, 64, 64).type(dtype)
    x_var = Variable(x.type(dtype))  # 需要将其封装为Variable类型。
    ans = fixed_model(x_var)

    print(np.array(ans.size()))  # 检查模型输出。
    np.array_equal(np.array(ans.size()), np.array([32, 10]))