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


FOLDERDATASET = os.getcwd()+"\static\dataset\*.png"

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

correloImages=[]
signatureImages = []
listeDesNom = []
signatureEtNom = []
for i in range(len(dataSet)):
    nom,sig,corr=correlogramme(dataSet[i])
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
        nom,sigreq,corReq=correlogramme(os.getcwd()+"/static/requete/"+str(msg)); 

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
        nomEtDistanceEuclidienne= nomEtDistanceEuclidienne [:6]
        
        return render(request, "search/search.html",{'form':form, 'msg':msg,'list':nomEtDistanceEuclidienne})
    else : 
        form=UploadFileForm()
        return render(request, "search/search.html",{'form':form})
   
def showHome(request):
    return render(request, 'search/home.html', {})

#i- Recherche par couleur : 
def getMoments(image):
    
    #extraction des composants RGB de l'image
    R=image[:,:,0] 
    G=image[:,:,1]
    B=image[:,:,2]
    
    #calcule des moyennes et des ecart-type de chaque composants
    colorFeature=[np.round(R.mean()),np.round(R.std()),np.round(G.mean()),np.round(G.std()),
                  np.round(B.mean()),np.round(B.std())]
    
    #normalisation
    colorFeature=colorFeature/np.array(colorFeature).mean()
    
    return colorFeature

def CIBR_Recherche(Folder,imageRequete):
    
    imReq_features=getMoments(imageRequete) #extraction du vecteur descripteur de l'image requete
    
     #dictionnaire qui va contenir les distances eucludiennes % a l'image requete et le nom des image de dataset
    dict_distance={}
    
    #lister les images contenues dans le dossier
    for imagePath in glob.glob(Folder+ "/*.png"):
        
        #recupere le nom de l'image courant
        image = cv.imread(imagePath) #lecture de l'image courant
        
        feature=getMoments(image) #recupere le vecetur descripteurs de l'image courant
        
        #calcule la distance euclidienne des vecteurs descripteurs de l'image courant du dataset et l'image requete
        
        distance= distance.euclidean(imReq_features,feature)
        
        
        #ajoute la distance eucludienne et la path de l'image dans comme cle-valeur dans le dictionnaire
        
        dict_distance.setdefault(imagePath,distance) 
        
    return dict_distance

def requete_RechercheCouleur(dataset,imageRequest,n=10):
    
    im_req=cv.imread(imageRequest) #lecture de l'image requete
    
    dict_res=CIBR_Recherche(dataset,imageRequete=im_req) 
    
    #trie du dictionnaire selon les distances eucludienne
    
    res_sorted=sorted(dict_res.items(),key = lambda x : x[1] )
    
    #recuperation des 5 premiere images les plus similaires
    
     
    res=res_sorted[:n]
    
    #affichage des ces n premieres image similaires
    
    fig=plt.figure(figsize=(20,20))
    plt.subplot(n-1,2,1)
    
    plt.imshow(im_req) #affichage de l'image Requete
    
    plt.title("Image Requete")
    plt.axis("off")
    
    i=3

    for key,value in res:
        im=cv.imread(key)
        
        plt.subplot(n-1,2,i)
        plt.imshow(im)
        plt.title("Image Similaire : "+key.split('\\')[0])
        plt.axis("off")
        i=i+1
        
        