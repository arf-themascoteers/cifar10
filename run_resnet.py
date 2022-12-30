import torch
import train
import test
from resnet_101_cifar import ResNet

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ",device)

    print("Training started...")
    model = ResNet()
    name = "resnet"
    train.train(device, model, name)

    print("Testing started...")
    test.test(device, name)