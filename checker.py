from torchvision import datasets, transforms
import torch
import os
from PIL import Image
import numpy as np

orig = "data/shifted/orig/0_0.png"
train_dir = "data/shifted/train/0_0.png"
image_orig = np.asarray(Image.open(orig))
image_train = np.asarray(Image.open(train_dir))

print(image_orig)
print(image_train)

print("done")