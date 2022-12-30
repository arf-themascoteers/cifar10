import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder


class DatasetManager:
    def __init__(self, train=True):
        torch.manual_seed(3)
        transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Resize((224,224)),
        ])
        self.dataset = datasets.CIFAR10(root="./data", train=train, transform=transform, download=True)

    def get_ds(self):
        return self.dataset


if __name__ == "__main__":
    dsm = DatasetManager()
    data_loader = torch.utils.data.DataLoader(dataset=dsm.get_ds(),batch_size=64,shuffle=True)
    for data, label in data_loader:
        print(data.shape)
        print(label.shape)
        print(label[0])
        for i in data:
            plt.imshow(i[0].numpy())
            plt.show()
            exit(0)
