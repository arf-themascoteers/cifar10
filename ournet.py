import torch
from modified_resnet import Bottleneck
from modified_resnet import ResNet


class OurNet(ResNet):
    def __init__(self):
        super().__init__(Bottleneck, [3, 4, 23, 3], filter_size=4)
        self.fc = torch.nn.Linear(1024, 10)


if __name__ == "__main__":
    model = OurNet()
    torch = torch.randn((1,3,32,32))
    out = model(torch)
    print(out.shape)
