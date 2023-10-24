import os
import torch
import torchvision
from .train import BEST_MODEL_PATH, DATA_DIR, DATA_TRANSFORMER
from .evaluate import evaluate


if __name__ == "__main__":
    # dataloader
    image_datasets = {
        x: torchvision.datasets.ImageFolder(os.path.join(DATA_DIR, x), DATA_TRANSFORMER)
        for x in ['test_T2', 'test_C0']
    }

    data_loaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16, shuffle=True)
        for x in ['test_T2', 'test_C0']
    }

    T2_DATA = data_loaders['test_T2']
    C0_DATA = data_loaders['test_C0']
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps')
    model = torch.load(BEST_MODEL_PATH, map_location=device)
    model.to(device)

    val_loss, val_acc = evaluate(model, T2_DATA, device)
    print(f"On T2 dataset loss={val_loss}, acc={val_acc}")
    val_loss, val_acc = evaluate(model, C0_DATA, device)
    print(f"On C0 dataset loss={val_loss}, acc={val_acc}")