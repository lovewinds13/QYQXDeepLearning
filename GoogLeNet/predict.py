"""
预测
"""

import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import GoogLeNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    image_path = "./daisy01.jpg"
    img = Image.open(image_path)
    plt.imshow(img)
    img = data_transform(img)   # [N, C H, W]
    img = torch.unsqueeze(img, dim=0)   # 维度扩展
    # print(f"img={img}")
    json_path = "./calss_indices.json"
    with open(json_path, 'r') as f:
        class_indict = json.load(f)

    # model = AlexNet(num_classes=5).to(device)   # GPU
    # model = vgg(model_name="vgg16", num_classes=5)  # CPU
    model = GoogLeNet(num_classes=5, aux_logits=False)
    weights_path = "./GoogLeNet.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    # model.load_state_dict(torch.load(weights_path))
    missing_keys, unexpected_keys = model.load_state_dict(torch.load(weights_path), strict=False)
    model.eval()    # 关闭 Dorpout
    with torch.no_grad():
        # output = torch.squeeze(model(img.to(device))).cpu()   #GPU
        output = torch.squeeze(model(img))      # 维度压缩
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
        print_res = "class: {}  prob: {:.3}".format(class_indict[str(predict_cla)],
                                                    predict[predict_cla].numpy())
        plt.title(print_res)
        # for i in range(len(predict)):
        #     print("class: {}  prob: {:.3}".format(class_indict[str(predict_cla)],
        #                                             predict[predict_cla].numpy()))
        plt.show()

if __name__ == '__main__':
    main()
