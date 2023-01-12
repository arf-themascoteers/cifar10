import torch
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
import model_resnet
from cifar import Cifar
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import cv2
import antialiased_cnns
import torch.nn as nn
import model_resnet
import model_mbp


def explore(device):
    cid = Cifar(train=False)
    model = torch.load(f"models/resnet.h5")
    model.eval()
    model.to(device)
    nodes, _ = get_graph_node_names(model)
    # print(nodes)
    # exit(0)
    x = cid[100][0]
    x = x.unsqueeze(dim=0)
    x = x.to(device)
    im = x[0].detach().cpu()
    im = im.permute(1, 2, 0)
    plt.imshow(im)
    plt.show()
    level = "maxpool"
    feature_extractor = create_feature_extractor(model, return_nodes=[level])

    out = feature_extractor(x)
    for count, i in enumerate(out[level][0]):
        plt.imshow(i.detach().cpu())
        plt.show()
        if count == 2:
            break
    exit(0)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    explore(device)
