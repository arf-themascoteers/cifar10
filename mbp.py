import torch.nn as nn
import torch.nn.functional as F
import antialiased_cnns


class MBP(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = antialiased_cnns.resnet101(pretrained=False)
        number_input = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(number_input, 10)
        )

    def forward(self, x):
        x = self.resnet(x)
        return F.log_softmax(x, dim=1)
