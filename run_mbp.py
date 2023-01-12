import torch
import train
import test
from mbp import MBP

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ",device)

    print("Training started...")
    model = MBP()
    name = "mbp"
    #train.train(device, model, name)

    print("Testing started...")
    test.test(device, name)