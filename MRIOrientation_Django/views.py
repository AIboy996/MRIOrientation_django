from django.shortcuts import render
from PIL import Image
import SimpleITK
import io
import urllib, base64
import tempfile
from .transform import ori

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
    return render(request, 'index.html', locals())

def loadimage(request):
    if request.method == 'POST' and request.FILES['nii_image']:
        nii_image = request.FILES['nii_image']
        with tempfile.NamedTemporaryFile(suffix=nii_image.name) as f:
            f.write(nii_image.file.read())
            file_name = f.name
            img_slices = read_ITK(file_name)
        uri_l = tobase64(img_slices)
        # TODO add predict algorithm
        uri_l_adjusted =  tobase64([ori(slice, '100') for slice in img_slices])
    return render(request, 'index.html', locals())