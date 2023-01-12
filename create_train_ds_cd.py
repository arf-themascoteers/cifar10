from torchvision import datasets, transforms
import torch
import os
from PIL import Image
import numpy as np
import shutil

torch.manual_seed(0)
transform = transforms.Compose([
    transforms.ToTensor(),
])

root_dir = "data/shifted/train"
source_dir = "data/catdog/train_transformed"

counter = [0,0]
for index, image_name in enumerate(os.listdir(source_dir)):
    image_path = os.path.join(source_dir, image_name)
    prefix = 0
    if image_name.startswith("dog"):
        prefix = 1
    dest_image_name = f"{prefix}_{counter[prefix]}.jpg"
    counter[prefix] = counter[prefix] + 1
    dest_image_path = os.path.join(root_dir, dest_image_name)
    shutil.copyfile(image_path, dest_image_path)

print("All done")