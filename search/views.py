from django.shortcuts import render
from .models import Predict
from .forms import UploadFileForm
import pathlib
import os
import glob
from skimage import io
import cv2 as cv 
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt
import imutils


FOLDERDATASET = os.getcwd()+"\static\dataset\*"

def sort_item(item):
    return item[1]

def readPictureDataSet () : 
    images = [file for file in glob.glob(FOLDERDATASET)]
    print(images)
    return images
dataSet = readPictureDataSet ()

#Création d'une fonction pour le calcul de corrélogramme : 
def correlogramme (emplacement): 
    sig =[0 for _ in range(256)]
    emplacement = str(emplacement)
    nom = emplacement[36:]
    image = io.imread(emplacement) 
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
    return nom,sig,corI1 

def cooccurence (emplacement) : 
    sig =[0 for _ in range(256)]
    # Load the image
    emplacement = str(emplacement)
    nom = emplacement[36:]
    image = io.imread(emplacement)
    imageGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Calculate the co-occurrence matrix
    cooc_matrix = np.zeros((256, 256))
    for i in range(imageGray.shape[0] - 1):
        for j in range(imageGray.shape[1] - 1):
            x = imageGray[i, j]
            y = imageGray[i, j + 1]
            cooc_matrix[x, y] += 1

    # Normalize the co-occurrence matrix
    cooc_matrix = cooc_matrix / np.sum(cooc_matrix)
    return nom,sig,cooc_matrix; 



correloImages=[]
signatureImages = []
listeDesNom = []
signatureEtNom = []
for i in range(len(dataSet)):
    nom,sig,corr=cooccurence(dataSet[i])
    correloImages.append(corr)
    for i in range(0, 256, 1) : 
       sig[i] = corr[i][i]
    signatureImages.append(sig) 
    listeDesNom.append(nom)
    signatureEtNom.append((nom,sig))
print(signatureEtNom[0][1]) 


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
        nom,sigreq,corReq=cooccurence(os.getcwd()+"/static/requete/"+str(msg)); 

        #II-Calcul de la signature 
        for i in range(0, 256, 1) : 
            sigreq[i] = corReq[i][i]

        #III- Calcule de la distance euclidienne de l'image requete avec images dans dataset : 
        nomEtDistanceEuclidienne = []
        for i in range(len(dataSet)): 
            distanceAvecRequete = distance.euclidean(sigreq, signatureEtNom[i][1])
            nom = signatureEtNom[i][0]
            nomEtDistanceEuclidienne.append((nom,distanceAvecRequete))
            print(nomEtDistanceEuclidienne)
        
        nomEtDistanceEuclidienne.sort(key=sort_item, reverse= False)
        print(nomEtDistanceEuclidienne)
        nomEtDistanceEuclidienne= nomEtDistanceEuclidienne [:10]
        
        return render(request, "search/search.html",{'form':form, 'msg':msg,'list':nomEtDistanceEuclidienne})
    else : 
        form=UploadFileForm()
        return render(request, "search/search.html",{'form':form})
   
def showHome(request):
    return render(request, 'search/home.html', {})