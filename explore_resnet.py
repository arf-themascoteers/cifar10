import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet101, ResNet101_Weights


resnet = resnet101(weights = None)

print(resnet)