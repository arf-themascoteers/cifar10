import torch
from antialiased_cnns.resnet import Bottleneck
from antialiased_cnns import ResNet


class MBP(ResNet):
    def __init__(self):
        super().__init__(Bottleneck, [3, 4, 23, 3], filter_size=4)
        self.fc = torch.nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        self.fx_x = x
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    model = MBP()
    torch = torch.randn((1,3,32,32))
    out = model(torch)
    print(out.shape)
