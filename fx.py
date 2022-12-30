import torch
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
import resnet
from dataset_manager import DatasetManager
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import cv2


def explore(device):
    cid = DatasetManager(train=False).get_ds()
    model = our_machine.ResNet()#torch.load("models/cnn_trans.h5")
    model.eval()
    model.to(device)
    nodes, _ = get_graph_node_names(model)
    feature_extractor = create_feature_extractor(model, return_nodes=['x', 'resnet.layer1.0.conv1'])
    x = cid[0][0]
    x = x.unsqueeze(dim=0)
    x = x.to(device)
    out = feature_extractor(x)
    print(out['resnet.layer1.0.conv1'].shape)
    out = out['resnet.layer1.0.conv1'][0]
    im = x[0].detach().cpu()
    im = im.permute(1, 2, 0)
    plt.imshow(im)
    plt.show()
    for count, i in enumerate(out):
        plt.imshow(i.detach().cpu())
        plt.show()
        if count == 3:
            break
    exit(0)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    explore(device)
