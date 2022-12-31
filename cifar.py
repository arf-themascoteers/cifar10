import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from PIL import Image


class Cifar(Dataset):
    def __init__(self, train=True):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.root = "data/shifted/train"
        if not train:
            self.root = "data/shifted/test"
        self.names = os.listdir(self.root)
        self.labels = [int(x.split(".")[0].split("_")[0]) for x in self.names]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.names[idx])
        image = Image.open(img_path)
        image = self.transform(image)
        label = self.labels[idx]
        return image, torch.tensor(label)


if __name__ == "__main__":
    dsm = Cifar()
    data_loader = torch.utils.data.DataLoader(dataset=dsm,batch_size=64,shuffle=True)
    for data, label in data_loader:
        print(data.shape)
        print(label.shape)
        print(label[0])
        for i in data:
            plt.imshow(i[0].numpy())
            plt.show()
            exit(0)
