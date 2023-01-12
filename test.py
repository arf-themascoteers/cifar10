import torch
from cifar import Cifar
from torch.utils.data import DataLoader


def test(device, name):
    batch_size = 1
    cid = Cifar(train=False)
    dataloader = DataLoader(cid, batch_size=batch_size, shuffle=True)
    model = torch.load(f"models/{name}.h5")
    model.eval()
    model.to(device)
    correct = 0
    total = 0

    true_ys = []
    pred_ys = []

    for (x, y) in dataloader:
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        pred = torch.argmax(y_hat, dim=1, keepdim=True)
        correct += pred.eq(y.data.view_as(pred)).sum()
        total += x.shape[0]

        for a_y in y:
            true_ys.append(a_y.detach().cpu().numpy())

        for a_y in pred:
            pred_ys.append(a_y[0].detach().cpu().numpy())

    print(f'Total:{total}, Correct:{correct}, Accuracy:{correct/total*100:.2f}')


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test(device)
