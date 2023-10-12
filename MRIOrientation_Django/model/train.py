import os
import torch
import torchvision
import torch.nn as nn
import torch.utils.data
from torchvision import transforms
import numpy as np

from .net import CNN
from .evaluate import evaluate


data_dir = r"C:\Users\yangz\Desktop\MRIOrientation\imgs"
# data_dir = "/Users/yang/Desktop/Medical Image Project/Orientation-Adjust-Tool/imgs"
BEST_MODEL_PATH = r'C:\Users\yangz\Desktop\MRIOrientation_django\MRIOrientation_django\model_best.pth'

# data augmentation
data_transformer = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop((256, 256), scale=(0.7, 1), ratio=(0.8, 1.2)),
        transforms.ToTensor()
])


if __name__ == "__main__":
    # dataloader
    image_datasets = {
        x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x), data_transformer)
        for x in ['train_data_LGE', 'valid_data_LGE', 'test_data_LGE']
    }

    data_loaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8, shuffle=True
                                    #    , num_workers=2
                                    )
        for x in ['train_data_LGE', 'valid_data_LGE', 'test_data_LGE']
    }

    train_iter = data_loaders['train_data_LGE']
    valid_iter = data_loaders['valid_data_LGE']
    test_iter = data_loaders['test_data_LGE']
    lr = 0.01
    num_epochs = 20
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps')
    model = CNN()
    model.to(device)
    best_loss = np.inf
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    # As stated in the torch.nn.CrossEntropyLoss() doc:
    # This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
    # Therefore, you should not use softmax before.

    for epoch in range(num_epochs):
        # train mode
        model.train()
        train_loss = 0
        for i, (X,y) in enumerate(train_iter):
            # Sets the gradients of all optimized torch.Tensor s to zero. 
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            l = loss(y_hat, y)
            # update param
            l.backward()
            optimizer.step()
            train_loss += l
        train_loss /= len(train_iter)
        val_loss = evaluate(model, valid_iter, device)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model, BEST_MODEL_PATH)
        print(f'epoch = {epoch+1}, {float(train_loss) = } {float(val_loss) = }')
    print(f'{float(best_loss) = }')

    best_model = torch.load(BEST_MODEL_PATH)
    test_loss = evaluate(model, test_iter, device)
    print(f'test loss on best model is {test_loss}')