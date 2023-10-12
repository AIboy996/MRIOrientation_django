import SimpleITK
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch.nn.functional import softmax

from .net import CNN
from .train import BEST_MODEL_PATH

model = torch.load(BEST_MODEL_PATH)
device = torch.device('cuda')
model.to(device)

def pred(img):
    X = torch.tensor(img, dtype=torch.float, device=device)
    X = X.reshape((1, 1, *X.shape))
    # resize to (256, 256)
    transformer = transforms.RandomResizedCrop(size=(256, 256), scale=(0.7, 1), ratio=(0.8, 1.2))
    X = transformer(X)
    outputs = model(X)
    # pred label
    pred = softmax(outputs, dim=1).argmax(axis=1)
    label = int(pred[0])
    print(f'pred label: {label}')
    return label


if __name__ == "__main__": # in Django is MRIOrientation_Django.model.predict
    print('patient1_LGE.nii.gz:')
    itk_img = SimpleITK.ReadImage('./data/patient1_LGE.nii.gz')
    img = SimpleITK.GetArrayFromImage(itk_img)
    for slice in img:
        print(pred(slice))

    print('sample.png:')
    img = Image.open('./data/sample.png').convert('L')
    img = np.array(img)
    print(pred(img))