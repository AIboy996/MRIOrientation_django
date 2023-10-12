from django.shortcuts import render
from PIL import Image
import SimpleITK
import io
import urllib, base64
import numpy as np
from .transform import ori
from .model.predict import pred

def read_ITK(file_name):
    itk_img = SimpleITK.ReadImage(file_name)
    img_slices = SimpleITK.GetArrayFromImage(itk_img)
    return img_slices

def tobase64(img_slices):
    uri_l = []
    for slice in img_slices:
        img = Image.fromarray(slice).convert('L')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri_l.append('data:image/png;base64,' + urllib.parse.quote(string))
    return uri_l

def index(request):
    file_name = './data/patient1_LGE.nii.gz'
    uri_l = tobase64(read_ITK(file_name))
    uri_l_adjusted = uri_l
    orientations = [0]*5
    orientations = [f'ori(slice{i+1}) = {orientation}' for i,orientation in enumerate(orientations)]
    return render(request, 'index.html', locals())

def loadimage(request):
    if request.method == 'POST' and request.FILES['nii_image']:
        nii_image = request.FILES['nii_image']
        with open(f'./received/received_{nii_image.name}', 'wb+') as f:
            f.write(nii_image.file.read())
            if nii_image.name.endswith('.nii.gz') or nii_image.name.endswith('.nii'):
                img_slices = read_ITK(f.name)
            elif nii_image.name.endswith('.png') or nii_image.name.endswith('.jpg'):
                img_slices = [np.array(Image.open(f.name).convert('L'))]
        uri_l = tobase64(img_slices)
        orientations = [pred(slice) for slice in img_slices]
        uri_l_adjusted =  tobase64([ori(slice, orientation) for slice, orientation in zip(img_slices,orientations)])
        orientations = [f'ori(slice{i+1}) = {orientation}' for i,orientation in enumerate(orientations)]
    return render(request, 'index.html', locals())