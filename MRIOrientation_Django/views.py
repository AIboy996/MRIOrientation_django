from django.shortcuts import render
from PIL import Image
import SimpleITK
import io
import urllib, base64

def ITKtobase64(file_name='./data/patient1_LGE.nii.gz'):
    itk_img = SimpleITK.ReadImage(file_name)
    img_slices = SimpleITK.GetArrayFromImage(itk_img)
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
    uri_l = ITKtobase64(file_name)
    return render(request, 'index.html', locals())