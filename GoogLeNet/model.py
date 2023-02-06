"""
VGG模型
"""


import torch.nn as nn
import torch
import torch.nn.functional as F


"""
# 定义卷积+激活函数操作模板
"""
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)

        return x

"""
# 定义 Iception 辅助分类器模板
"""
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)    # output = [batch, 128, 4, 4]
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N*512*14*14, aux2: N*528*14*14
        x = self.averagePool(x)
        # aux1: N*512*4*4, aux2: N*528*4*4
        x = self.conv(x)
        # N*128*4*4
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=self.training)
        # N*2048
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)
        # N*1024
        x = self.fc2(x)
        # N*num_classes
        return x

"""
# Inception 模板 
"""
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)    # 保证输出大小等于输入大小
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            # 官方 3x3, https://github.com/pytorch/vision/issues/906
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)  # 输出大小=输入大小
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]

        return torch.cat(outputs, 1)    # 拼接数据

"""
# GoogLeNet 模型
"""
class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N*3*224*224
        x = self.conv1(x)
        # N*64*112*112
        x = self.maxpool1(x)
        # N*64*56*56
        x = self.conv2(x)
        # N*64*56*56
        x = self.conv3(x)
        # N*192*56*56
        x = self.maxpool2(x)

        # N*192*28*28
        x = self.inception3a(x)
        # N*256*28*28
        x = self.inception3b(x)
        # N*480*28*28
        x = self.maxpool3(x)
        # N*480*14*14
        x = self.inception4a(x)
        # N*512*14*14
        if self.training and self.aux_logits:  # eval model lose this layer
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N*512*14*14
        x = self.inception4c(x)
        # N*512*14*14
        x = self.inception4d(x)
        # N*528*14*14
        if self.training and self.aux_logits:  # eval model lose this layer
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N*832*14*14
        x = self.maxpool4(x)
        # N*832*7*7
        x = self.inception5a(x)
        # N*832*7*7
        x = self.inception5b(x)
        # N*1024*7*7

        x = self.avgpool(x)
        # N*1024*1*1
        x = torch.flatten(x, 1)
        # N*1024
        x = self.dropout(x)
        x = self.fc(x)
        # N*1000 (num_classes)
        if self.training and self.aux_logits:  # eval model lose this layer
            return x, aux2, aux1

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

"""
测试模型
"""
# if __name__ == '__main__':
#     input1 = torch.rand([224, 3, 224, 224])
#     model_x = GoogLeNet(num_classes=5, aux_logits=True, init_weights=True)
#     print(model_x)
    # output = GoogLeNet(input1)
