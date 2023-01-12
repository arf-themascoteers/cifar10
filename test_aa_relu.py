import matplotlib.pyplot as plt
from aa_relu import AA_ReLU
import torch

model = AA_ReLU()
model.train()
print(model.alpha)
y = torch.nn.functional.relu(torch.linspace(-1,1,steps=100))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
x = torch.linspace(-100,100,steps=100)
out = model(x)
print(out)
plt.plot(out.detach().numpy())
plt.show()
for i in range(10):
    x = torch.linspace(-1,1,steps=100)
    optimizer.zero_grad()
    out = model(x)
    loss = torch.nn.functional.mse_loss(out, y)
    loss.backward()
    optimizer.step()

x = torch.linspace(-1,1,steps=100)
out = model(x)
print(out)
plt.plot(out.detach().numpy())
plt.show()
