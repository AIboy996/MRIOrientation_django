import os
import torch
import torchvision
import torch.nn as nn
import torch.utils.data
from torchvision import transforms
import numpy as np
from net import CNN
from evaluate import evaluate

data_dir = "/Users/yang/Desktop/Orientation-Adjust-Tool/imgs"
BEST_MODEL_PATH = './model_best.pth'

# data augmentation
data_transformer = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop((256, 256), scale=(0.7, 1), ratio=(0.8, 1.2)),
        transforms.ToTensor()
])
# dataloader
image_datasets = {
    x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x), data_transformer)
    for x in ['train_data_LGE', 'valid_data_LGE', 'test_data_LGE']
}
data_loaders = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8, shuffle=True, num_workers=2)
    for x in ['train_data_LGE', 'valid_data_LGE', 'test_data_LGE']
}

train_iter = data_loaders['train_data_LGE']
valid_iter = data_loaders['valid_data_LGE']
test_iter = data_loaders['test_data_LGE']

lr = 0.01
num_epochs = 20
device = torch.device('mps')
best_loss = np.inf
model = CNN()

model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
loss = nn.CrossEntropyLoss()
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for i, (X,y) in enumerate(train_iter):
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)
        y_hat = model(X)
        l = loss(y_hat, y)
        l.backward()
        optimizer.step()
        train_loss += l
    train_loss /= len(train_iter)
    val_loss = evaluate(model, valid_iter, device)
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model, BEST_MODEL_PATH)
    print(f'{epoch = }, {train_loss = } {val_loss = }')
