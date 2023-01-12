import torch
import train
import test
from ournet import OurNet

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ",device)

    print("Training started...")
    model = OurNet()
    name = "our"
    train.train(device, model, name)

    print("Testing started...")
    test.test(device, name)