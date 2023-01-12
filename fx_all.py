import torch
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
import model_resnet
from cifar import Cifar
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import cv2
import antialiased_cnns
import torch.nn as nn
import model_mbp
import model_mbp


def explore(device, model, x):
    model = torch.load(f"models/{model}.h5")
    model.eval()
    model.to(device)
    x = x.to(device)
    model(x)
    fx_x = model.fx_x
    img = fx_x[0][0]
    img = img.detach().cpu()
    img = img/img.max()
    # img[img<0.5]=0
    # img[img>=0.5]=1
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = ["resnet", "mbp", "our"]
    cid = Cifar(train=False)
    for i in range(5):
        x = cid[i][0]
        x = x.unsqueeze(dim=0)

        im = x[0].detach().cpu()
        im = im.permute(1, 2, 0)
        plt.imshow(im)
        plt.show()

        for model in models:
            explore(device, model, x)
