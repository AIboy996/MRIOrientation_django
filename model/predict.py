import torch
import torch.nn as nn
import SimpleITK

state_dict = torch.load('../model/model_0912.pth', map_location=torch.device('mps'))
model.load_state_dict(state_dict)
model.eval()

itk_img = SimpleITK.ReadImage('../data/patient1_LGE.nii.gz')
img_slices = SimpleITK.GetArrayFromImage(itk_img)
X = torch.tensor(img_slices[0])
print(model(X))