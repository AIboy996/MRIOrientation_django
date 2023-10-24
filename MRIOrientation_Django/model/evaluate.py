import torch
import torch.nn as nn

loss = nn.CrossEntropyLoss()

@torch.inference_mode()
def evaluate(model, dataset, device):
    model.eval()
    num_val_batches = len(dataset)
    train_loss = 0
    train_acc = 0
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=True):
        for i, (X,y) in enumerate(dataset):
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            l = loss(y_hat, y)
            train_acc += torch.sum(y_hat.argmax(axis=1)==y)/y.shape[0]
            train_loss += l
    return train_loss/num_val_batches, train_acc/num_val_batches