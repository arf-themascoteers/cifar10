from torch import Tensor
import torch
import torchvision
from torchvision.models.resnet import Bottleneck
from torch import flatten


class ResNet101_CIFAR(torchvision.models.ResNet):
    def __init__(self):
        super().__init__(Bottleneck, [3, 4, 23, 3])
        self.fc = torch.nn.Linear(1024, 10)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #x = self.layer4(x)

        x = self.avgpool(x)
        x = flatten(x, 1)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    model = ResNet101_CIFAR()
    torch = torch.randn((1,3,32,32))
    out = model(torch)
    print(out.shape)
