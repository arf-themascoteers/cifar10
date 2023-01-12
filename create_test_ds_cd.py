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
dest_dir = "data/shifted/test"

counter = [0,0]
for index, image_name in enumerate(os.listdir(root_dir)):
    image_path = os.path.join(root_dir, image_name)
    dest_image_path = os.path.join(dest_dir, image_name)

    image_orig_pil = Image.open(image_path)
    image_orig = np.asarray(image_orig_pil)
    image = np.copy(image_orig)

    first_row = np.copy(image[0, :, :])
    image[0:-1, :, :] = image[1:, :, :]
    image[-1, :, :] = first_row
    first_col = np.copy(image[:, 0, :])
    image[:, 0:-1, :] = image[:, 1:, :]
    image[:, -1, :] = first_col

    Image.fromarray(image).save(dest_image_path)

    if index % 100 == 0:
        print(f"Done {index}")


print("All done")