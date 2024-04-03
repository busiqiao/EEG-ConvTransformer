import torch


def train(model, optimizer, x, y):
    model.train()
    optimizer.zero_grad()
    y_ = model(x)
    loss = torch.nn.functional.cross_entropy(y_, y)
    loss.backward()
    optimizer.step()
    return loss, y_


def test(model, x, y):
    x = x.cuda()
    y = y.cuda()
    model.eval()
    y_ = model(x)
    loss = torch.nn.functional.cross_entropy(y_, y)
    corrects = (torch.argmax(y_, dim=1).data == y.data)
    acc = corrects.cpu().int().sum().numpy()
    return loss, acc
