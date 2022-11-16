""""
测试
"""
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import LeNet


def main():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_class = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()
    net.load_state_dict(torch.load('LeNet.pth'))
    # net.load_state_dict(torch.load('LeNet.pth', map_location=torch.device("cpu")))

    test_image = Image.open('cat_test2.jpg')
    test_image = transform(test_image)  # [C H W]
    test_image = torch.unsqueeze(test_image, dim=0)  # [N C H W]

    with torch.no_grad():
        outputs = net(test_image)
        predict = torch.max(outputs, dim=1)[1].numpy()
    print(f"It is {data_class[int(predict)]}")

if __name__ == '__main__':
    main()
