"""
模型
"""


import torch.nn as nn
import torch


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        """
        特征提取
        """
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),   # 输入[3, 224, 224] 输出[48, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 输出 [48,27,27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),   # 输出 [128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 输出 [128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # 输出[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1), # 输出[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # 输出[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)   # 输出 [128, 6, 6]
        )
        """
        分类器
        """
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),  # Dropout 随机失活神经元, 比例诶0.5
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)

        return x

    """
    权重初始化
    """
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.01)
                nn.init.constant_(m.bias, 0)

"""
测试模型
"""
# if __name__ == '__main__':
#     input1 = torch.rand([224, 3, 224, 224])
#     model_x = AlexNet()
#     print(model_x)
    # output = AlexNet(input1)
