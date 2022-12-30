import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet101, ResNet101_Weights


class OurMachine(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet101(weights = None)
        number_input = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(number_input, 10)
        )

    def forward(self, x):
        x = self.resnet(x)
        return F.log_softmax(x, dim=1)
