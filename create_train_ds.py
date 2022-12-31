from torchvision import datasets, transforms
import torch
import os
from PIL import Image
import numpy as np

torch.manual_seed(0)
transform = transforms.Compose([
    transforms.ToTensor(),
])

counter = np.zeros(10, dtype=np.int32)

root_dir = "data/shifted/train"
if not os.path.exists(root_dir):
    os.mkdir(root_dir)

for train in [True, False]:
    dataset = datasets.CIFAR10(root="./data", train=True, download=True)
    for index, data in enumerate(dataset):
        x = data[0]
        y = data[1]
        count_item = counter[y]
        name = f"{y}_{count_item}.png"
        location = os.path.join(root_dir, name)
        counter[y] = count_item + 1

        image_orig = np.asarray(x)
        Image.fromarray(image_orig).save(location)
        if index%1000 == 0:
            print(f"Done {index}")



print("All done")