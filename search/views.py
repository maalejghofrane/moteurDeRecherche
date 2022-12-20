from django.shortcuts import render
from .models import Predict
from .forms import UploadFileForm
import pathlib
import os

def upload_file(request):
    if request.method=="POST":
    # récupérer le chemin du répertoire courant
        print("Le répertoire courant est : " + os.getcwd())
        form = UploadFileForm(request.POST,request.FILES)
        file = request.FILES['file']
        document = Predict.objects.create(name='requete', file=file)
        document.save()
        DATADIR = file
        data_dir = pathlib.Path(str(DATADIR))
        msg=data_dir
        list =['chat.jpg','chat.jpg']
        return render(request, "search/search.html",{'form':form, 'msg':msg,'list':list})
    else : 
        form=UploadFileForm()
        return render(request, "search/search.html",{'form':form})
   
def showHome(request):
    return render(request, 'search/home.html', {})