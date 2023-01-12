import torch
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
import resnet_101_cifar
from cifar import Cifar
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import cv2
import antialiased_cnns
import torch.nn as nn
import mbp
import mbp


def explore(device):
    cid = Cifar(train=False)
    model = torch.load(f"models/our.h5")
    model.eval()
    model.to(device)
    x = cid[100][0]
    x = x.unsqueeze(dim=0)
    x = x.to(device)
    im = x[0].detach().cpu()
    im = im.permute(1, 2, 0)
    plt.imshow(im)
    plt.show()
    out = model(x)
    fx_x = model.fx_x
    for count, i in enumerate(fx_x[0]):
        plt.imshow(i.detach().cpu())
        plt.show()
        if count == 2:
            break
    exit(0)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    explore(device)
