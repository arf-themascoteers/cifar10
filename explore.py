from cifar import Cifar
import torch

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = Cifar(train=True)[0][0]
    test = Cifar(train=False)[0][0]
    print("")