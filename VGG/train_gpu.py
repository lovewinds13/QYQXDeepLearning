"""
训练(GPU)
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
from tqdm import tqdm

from model import vgg


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"use device is {device}")

    data_transform = {
        "train": transforms.Compose([
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
        "val": transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    }
    # data_root = os.path.abspath(os.path.join(os.getcwd(), "../..")) # 读取数据路径
    data_root = os.path.abspath(os.path.join(os.getcwd(), "./"))
    image_path = os.path.join(data_root, "data_set", "flower_data")
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

    model_name = "vgg16"
    net = vgg(model_name=model_name, num_classes=5, init_weights=True)  # 实例化网络(5分类)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    epochs = 10
    save_path = "./VGGNet_GPU.pth"
    best_accuracy = 0.0
    train_steps = len(train_loader)
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch [{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                      epochs,
                                                                      loss
                                                                      )
        # 验证
        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
        val_accuracy = acc / val_num
        print("[epoch %d ] train_loss: %3f    val_accurancy: %3f" %
              (epoch + 1, running_loss / train_steps, val_accuracy))
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(net.state_dict(), save_path)
    print("Finshed Training.")

if __name__ == '__main__':
    main()

