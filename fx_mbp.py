import torch
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
import resnet
from dataset_manager import DatasetManager
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import cv2
import antialiased_cnns
import torch.nn as nn
import mbp
import resnet


def explore(device):
    cid = DatasetManager(train=False).get_ds()
    model = antialiased_cnns.resnet101(pretrained=False)
    model.layer2 = nn.Sequential(
        model.layer2,
        nn.Flatten()
    )
    print(model)
    model.eval()
    model.to(device)
    x = cid[3][0]
    x = x.unsqueeze(dim=0)
    x = x.to(device)
    # im = x[0].detach().cpu()
    # im = im.permute(1, 2, 0)
    # plt.imshow(im)
    # plt.show()
    out = model(x)
    print(out.shape)
    # for count, i in enumerate(out):
    #     plt.imshow(i.detach().cpu())
    #     plt.show()
    #     if count == 3:
    #         break
    exit(0)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    explore(device)
