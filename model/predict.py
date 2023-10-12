import torch
from net import CNN
import SimpleITK
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.nn.functional import softmax
model = torch.load('./model_best.pth')
device = torch.device('cuda')
model.to(device)

if __name__ == "__main__":
    # itk_img = SimpleITK.ReadImage('../data/patient1_LGE.nii.gz')
    # img = SimpleITK.GetArrayFromImage(itk_img)[0]
    img = Image.open('../data/sample.png').convert('L')
    img = np.array(img)
    X = torch.tensor(img, dtype=torch.float, device=device)
    X = X.reshape((1, 1, *X.shape))
    # resize to (256, 256)
    transformer = transforms.RandomResizedCrop(size=(256, 256), scale=(0.7, 1), ratio=(0.8, 1.2))
    X = transformer(X)
    outputs = model(X)
    # pred label
    pred = softmax(outputs, dim=1).argmax(axis=1)
    print(int(pred[0]))