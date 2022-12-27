from django.shortcuts import render
from .models import Predict
from .forms import UploadFileForm
import pathlib
import os
import glob
from skimage import io
import cv2 as cv 
from scipy.spatial import distance

def sort_item(item):
    return item[2]
#Création d'une fonction pour le calcul de corrélogramme : 
def correlogramme (emplacement): 
    emplacement = str(emplacement)
    image = io.imread(os.getcwd()+emplacement) 
    imageGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    corI1 = [[1 for _ in range(256)] for _ in range(256)]
    sig1 =[0 for _ in range(256)]
    distance1=0
    # Calcule corrélogramme de l'image 1 avec distance d=1 : 
    for x in range(0, 239, 1) : 
        for y in range(0, 239, 1) : 
            o = imageGray[x][y]
            o1=imageGray[x-1][y-1]
            corI1[o][o1]=corI1[o][o1]+1
            o2=imageGray[x-1][y]
            corI1[o][o2]=corI1[o][o2]+1
            o3=imageGray[x-1][y+1]
            corI1[o][o3]=corI1[o][o3]+1
            o4=imageGray[x][y-1]
            corI1[o][o4]=corI1[o][o4]+1
            o6=imageGray[x][y+1]
            corI1[o][o6]=corI1[o][o6]+1
            o7=imageGray[x+1][y-1]
            corI1[o][o7]=corI1[o][o7]+1
            o8=imageGray[x+1][y]
            corI1[o][o8]=corI1[o][o8]+1
            o9=imageGray[x+1][y+1]
            corI1[o][o9]=corI1[o][o9]+1
    return sig1,distance1,corI1 

def readPictureDataSet () : 
    images = [cv.imread(file) for file in glob.glob("/static/dataset/*.png")]
    return images
 

#0-Calculer le corrèlogramme de toute la dataset  
def readPictureDataSet () : 
    images = [file for file in glob.glob("C:\moteurDeRecherche\static\dataset\*.png")]
    return images
dataSet = readPictureDataSet ()
# correloImages=[]
# signatureImages = []
# for i in range(len(dataSet)):
#     sig1,corrI1=correlogramme(dataSet[i])
#     correloImages.append(corr)
#     for i in range(0, 256, 1) : 
#       sig1[i] = corrI1[i][i]
#     signatureImages.append(sig1)  
      

sig1,distance1,corI1=correlogramme("/static/dataset/image1.png"); 
sig2,distance2,corI2=correlogramme("/static/dataset/image2.png"); 
sig3,distance3,corI3=correlogramme("/static/dataset/image3.png"); 

def upload_file(request):
    if request.method=="POST":
        print("Le répertoire courant est : " + os.getcwd())
        form = UploadFileForm(request.POST,request.FILES)
        file = request.FILES['file']
        document = Predict.objects.create(name='requete', file=file)
        document.save()
        DATADIR = file
        data_dir = pathlib.Path(str(DATADIR))
        msg=data_dir
        #I-Calcul correlogramme de la requete 
        sigreq,distanceReq,corReq=correlogramme("/static/requete/"+str(msg)); 

        #II-Calcul de la signature 
        for i in range(0, 256, 1) : 
            sig1[i] = corI1[i][i]
            sig2[i] = corI2[i][i]
            sig3[i] = corI3[i][i]
            sigreq[i] = corReq[i][i]

        #III- Calcule de la distance euclidienne de l'image requete avec images dans dataset : 
        distanceRequeteImage1 = distance.euclidean(sigreq, sig1)
        distanceRequeteImage2 = distance.euclidean(sigreq, sig2)
        distanceRequeteImage3 = distance.euclidean(sigreq, sig3)

        listDistancesEuclidienne = [('image1.png','distanceEuclidienne1',distanceRequeteImage1),
                                    ('image2.png','distanceEuclidienne2',distanceRequeteImage2),
                                    ('image3.png','distanceEuclidienne3',distanceRequeteImage3)]
        listDistancesEuclidienne.sort(key=sort_item, reverse= False)
        print(listDistancesEuclidienne)
                                    
        #IV-Récupérer la liste des images 'listImage' 
        # et la liste de leurs signatures 'listeSignatureEuclidinne'
        listImage = []
        for i in listDistancesEuclidienne:
            listImage.append(i[0])
    
        listeSignatureEuclidinne = []
        for i in listDistancesEuclidienne:
            listeSignatureEuclidinne.append(i[2])
        return render(request, "search/search.html",{'form':form, 'msg':msg,'list':listImage,'listeSignatureEuclidinne':listeSignatureEuclidinne})
    else : 
        form=UploadFileForm()
        return render(request, "search/search.html",{'form':form})
   
def showHome(request):
    return render(request, 'search/home.html', {})