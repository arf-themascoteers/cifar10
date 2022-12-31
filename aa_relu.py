from torch import Tensor
import torch
from torch import nn
from math import exp, pi
from torch import sin,log
import torch.nn.functional as F


class AA_ReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([6],dtype=torch.float32))
        self.alpha.requires_grad = True

    def forward(self, x):
        x[x<=0] = 0
        mask = (self.alpha < x) & (x < self.alpha * exp(pi / 2))
        x[mask] = self.alpha * sin(log(x[mask] / self.alpha)) + self.alpha
        x[x > self.alpha * exp(pi / 2)] = 2 * self.alpha
        return x


if __name__ == "__main__":
    model = AA_ReLU()
    model.train()
    print(model.alpha)
    t2 = torch.randn((1,10))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
    for i in range(10):
        t1 = torch.randn((1, 10))
        optimizer.zero_grad()
        out = model(t1)
        loss = F.mse_loss(out, t2)
        loss.backward()
        optimizer.step()

    print(model.alpha)