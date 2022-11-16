"""
模型
"""


import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module): # 集成nn.Module父类
    def __init__(self):
        super(LeNet, self).__init__()

        # 看一下具体的参数
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=16,
                               kernel_size=5,
                               stride=1,
                               padding=0,
                               bias=True
                               )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # self.relu = nn.ReLU(inplace=True)

    # 正向传播
    def forward(self, x):
        x = F.relu(self.conv1(x))   # 输入: (3, 32, 32), 输出: (16, 28, 28)
        x = self.pool1(x)   # 输出: (16, 14, 14)
        x = F.relu(self.conv2(x))   # 输出: (32, 10, 10)
        x = self.pool2(x)   # 输出: (32, 5, 5)
        x = x.view(-1, 32*5*5)  # 输出: (32*5*5)
        x = F.relu(self.fc1(x))     # 输出: (120)
        x = F.relu(self.fc2(x))     # 输出: (84)
        x = self.fc3(x)     # 输出(10)

        return x

# """
# 调试信息, 查看模型参数传递
# """
# import torch
# input1 = torch.rand([32, 3, 32, 32])
# modelx = LeNet()
# print(modelx)
# output = modelx(input1)
