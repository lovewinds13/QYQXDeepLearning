"""
训练
"""
import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),  # 数据转为张量
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化处理
    ])
    # 导入训练集数据(50000张图片)
    train_set = torchvision.datasets.CIFAR10(root='./data', # root: 数据集存储路径
                                             train=True,    # 数据集为训练集
                                             download=False,  # download: True时下载数据集(下载完成修改为False)
                                             transform=transform    # 数据预处理
                                             )
    #   加载训练集
    train_loader = torch.utils.data.DataLoader(train_set,   # 加载训练集
                                               batch_size=50,   # batch 大小
                                               shuffle=True,    # 是否随机打乱训练集
                                               num_workers=0    # 使用的线程数量
                                               )
    # 导入测试集(10000张图片)
    val_set = torchvision.datasets.CIFAR10(root='./data',
                                           train=False,     # 数据集为测试集
                                           download=False,
                                           transform=transform
                                           )
    # 加载测试集数据
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=10000,   # 测试集batch大小
                                             shuffle=False,
                                             num_workers=0
                                             )
    # 获取测试集中的图片和标签
    val_data_iter = iter(val_loader)
    # val_image, val_label = val_data_iter.next()
    val_image, val_label = next(val_data_iter)  #python 3

    """
    # -------------------------------------------------------------------------------------------
    查看数据集, 注意修改查看数据集的 batch
    """
    # 定义的分类标签
    # class_labels = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
   # 查看数据集的图片
   #  def img_show(img):
   #      img = img / 2 + 0.5
   #      np_img = img.numpy()
   #      plt.imshow(np.transpose(np_img, (1, 2, 0)))
   #      plt.show()
   #
   #  # 查看数据集中的5张图像
   #  print(''.join(" %5s " % class_labels[val_label[j]] for j in range(5)))
   #  img_show(torchvision.utils.make_grid(val_image))
    """
    # -------------------------------------------------------------------------------------------
    """

    # 检查是否支持CPU
    # if torch.cuda.is_available():
    #     use_dev = torch.device("cuda")
    # else:
    #     use_dev = torch.device("cpu")
    # print(use_dev)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = LeNet()   # 用于训练的网络模型
    # 指定GPU or CPU 进行训练
    net.to("cpu")
    loss_function = nn.CrossEntropyLoss()   # 损失函数(交叉熵函数)
    optimizer = optim.Adam(net.parameters(), lr=0.001)  # 优化器(训练参数, 学习率)

    # 训练的轮数
    for epoch in range(5):
        start_time = time.perf_counter()
        running_loss = 0.0
        # 遍历训练集, 从0开始
        for step, data in enumerate(train_loader, start=0):
            inputs, labels = data   # 得到训练集图片和标签
            optimizer.zero_grad()   # 清除历史梯度
            outputs = net(inputs)   # 正向传播
            loss = loss_function(outputs, labels)   # 损失计算
            loss.backward() # 反向传播
            optimizer.step()    #优化器更新参数

            # 用于打印精确率等评估参数
            running_loss += loss.item()
            if step % 500 == 499:   # 500步打印一次
                with torch.no_grad():
                    outputs = net(val_image)    # 传入测试集数据
                    predict_y = torch.max(outputs, dim=1)[1]
                    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)

                    # 打印训练轮数、精确率等
                    print("[%d, %5d] train_loss: %.3f   test_accuracy: %.3f" %
                          (epoch + 1, step + 1, running_loss / 500, accuracy)
                          )
                    running_loss = 0.0
        end_time = time.perf_counter()
        print("cost time = ", end_time - start_time)

    print("Finished trainning")

    save_path = "./LeNet.pth"
    torch.save(net.state_dict(), save_path) # 保存训练输出的模型文件

if __name__ == '__main__':
    main()
