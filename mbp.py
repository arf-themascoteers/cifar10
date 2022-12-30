import torch.nn as nn
import torch.nn.functional as F
import antialiased_cnns
from torchvision.models import resnet101


class MBP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = antialiased_cnns.resnet101(pretrained=False)
        number_input = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(number_input, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    model = MBP()
    print(model)
