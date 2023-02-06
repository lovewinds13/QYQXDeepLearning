"""
训练(CPU)
"""
import os
import sys
import json
import time
import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm   # 显示进度条模块

from model import GoogLeNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"use device is {device}")

    data_transform = {
        "train": transforms.Compose([
                                    transforms.RandomResizedCrop(224),  # 随机裁剪, 再缩放为 224*224
                                    transforms.RandomHorizontalFlip(),  # 水平随机翻转
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
        "val": transforms.Compose([
                                    transforms.Resize((224, 224)),  # 元组(224, 224)
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    }
    # data_root = os.path.abspath(os.path.join(os.getcwd(), "../..")) # 读取数据路径
    data_root = os.path.abspath(os.path.join(os.getcwd(), "./"))
    image_path = os.path.join(data_root, "data_set", "flower_data")
    # print(image_path)
    # image_path = data_root + "/data_set/flower_data/"
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"]
                                         )
    train_num = len(train_dataset)
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open("calss_indices.json", 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # 线程数计算
    nw = 0
    print(f"Using {nw} dataloader workers every process.")

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw
                                               )
    val_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                       transform=data_transform["val"]
                                       )
    val_num = len(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=4,
                                             shuffle=False,
                                             num_workers=nw
                                             )
    print(f"Using {train_num} images for training, {val_num} images for validation.")

    # test_data_iter = iter(val_loader)
    # test_image, test_label = next(test_data_iter)

    """ 测试数据集图片"""
    # def imshow(img):
    #     img = img / 2 + 0.5
    #     np_img = img.numpy()
    #     plt.imshow(np.transpose(np_img, (1, 2, 0)))
    #     plt.show()
    # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(test_image))

    net = GoogLeNet(num_classes=5, aux_logits=True, init_weights=True)  # 实例化网络(5分类)
    # net.to(device)
    net.to("cpu")   # 直接指定 cpu
    loss_function = nn.CrossEntropyLoss()   # 交叉熵损失
    optimizer = optim.Adam(net.parameters(), lr=0.0003)     # 优化器(训练参数, 学习率)

    epochs = 10     # 训练轮数
    save_path = "./GoogLeNet.pth"
    best_accuracy = 0.0
    train_steps = len(train_loader)
    for epoch in range(epochs):
        net.train()     # 开启Dropout
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)     # 设置进度条图标
        for step, data in enumerate(train_bar):     # 遍历训练集,
            images, labels = data   # 获取训练集图像和标签
            optimizer.zero_grad()   # 清除历史梯度
            logits, aux_logits2, aux_logits1 = net(images)
            loss0 = loss_function(logits, labels)
            loss1 = loss_function(aux_logits1, labels)
            loss2 = loss_function(aux_logits2, labels)
            loss = loss0 + loss1 * 0.3 + loss2 * 0.3   # 计算损失值
            loss.backward()     # 方向传播
            optimizer.step()    # 更新优化器参数
            running_loss += loss.item()
            train_bar.desc = "train epoch [{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                      epochs,
                                                                      loss
                                                                      )
        # 验证
        net.eval()      # 关闭Dropout
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels).sum().item()
        val_accuracy = acc / val_num
        print("[epoch %d ] train_loss: %3f    val_accurancy: %3f" %
              (epoch + 1, running_loss / train_steps, val_accuracy))
        if val_accuracy > best_accuracy:    # 保存准确率最高的
            best_accuracy = val_accuracy
            torch.save(net.state_dict(), save_path)
    print("Finshed Training.")

if __name__ == '__main__':
    main()

