from django.shortcuts import render
from .models import Predict
from .forms import UploadFileForm
import pathlib
import os
from skimage import io
import cv2 as cv 
from scipy.spatial import distance


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

sig1,distance1,corI1=correlogramme("/static/dataset/image1.png"); 
sig2,distance2,corI2=correlogramme("/static/dataset/image2.png"); 
sig3,distance3,corI3=correlogramme("/static/dataset/image3.png"); 

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
        sigreq,distance10,corI10=correlogramme("/static/requete/"+str(msg)); 
        
        # #Lire Image Requete : 
        # image0 = io.imread(os.getcwd()+'/static/requete/'+str(msg)) 
        # image0Gray = cv.cvtColor(image0, cv.COLOR_BGR2GRAY)
        # print(image0Gray)
        # corI10 = [[1 for _ in range(256)] for _ in range(256)]
        # sigreq =[0 for _ in range(256)]
        # distance10=0

        # # Calcule corrélogramme de l'image Requete : 
        # for x in range(0, 239, 1) : 
        #     for y in range(0, 239, 1) : 
        #         o = image0Gray[x][y]
        #         o1=image0Gray[x-1][y-1]
        #         corI10[o][o1]=corI10[o][o1]+1
        #         o2=image0Gray[x-1][y]
        #         corI10[o][o2]=corI10[o][o2]+1
        #         o3=image0Gray[x-1][y+1]
        #         corI10[o][o3]=corI10[o][o3]+1
        #         o4=image0Gray[x][y-1]
        #         corI10[o][o4]=corI10[o][o4]+1
        #         o6=image0Gray[x][y+1]
        #         corI10[o][o6]=corI10[o][o6]+1
        #         o7=image0Gray[x+1][y-1]
        #         corI10[o][o7]=corI10[o][o7]+1
        #         o8=image0Gray[x+1][y]
        #         corI10[o][o8]=corI10[o][o8]+1
        #         o9=image0Gray[x+1][y+1]
        #         corI10[o][o9]=corI10[o][o9]+1

        #Calcul de la signature 
        for i in range(0, 256, 1) : 
            sig1[i] = corI1[i][i]
            sig2[i] = corI2[i][i]
            sig3[i] = corI3[i][i]
            sigreq[i] = corI10[i][i]

        #Calcule de la distance euclidienne : 
        d10vs1 = distance.euclidean(sigreq, sig1)
        d10vs2 = distance.euclidean(sigreq, sig2)
        d10vs3 = distance.euclidean(sigreq, sig3)

        #Initialisation des distances : 
        distance1=0
        distance2=0
        distance3=0

        #Calcule distancelokhra :
        for i in range(0, 256, 1) : 
         distance1=distance1+min(sigreq[i],sig1[i])
        d10Inter1 = abs(1-(distance1/240**2))
        print(d10Inter1)

        for i in range(0, 256, 1) : 
            distance2=distance2+min(sigreq[i],sig1[i])
        d10Inter2 = abs(1-(distance1/240**2))
        print(d10Inter2)

        for i in range(0, 256, 1) : 
            distance3=distance3+min(sigreq[i],sig1[i])
            d10Inter3 = abs(1-(distance1/240**2))
            print(d10Inter3)

        #liste distance lokhra : 
        listDistances = [('image1.png','distance1',d10Inter1),
                         ('image2.png','distance2',d10Inter2),
                         ('image3.png','distance3',d10Inter3)]
        print(listDistances[0])

        #Liste distance euclidienne : 
        listDistancesEuclidienne = [('image1.png','distanceEuclidienne1',d10vs1),
                                    ('image2.png','distanceEuclidienne2',d10vs2),
                                    ('image3.png','distanceEuclidienne3',d10vs3)]
        print(listDistancesEuclidienne[0][0])

        dataSet = ['image1.png','image2.png','image3.png','image4.png','image5.png','image6.png',
                    'image7.png','image8.png','image9.png','image10.png']

        def sort_item(item):
            return item[2]

        #Trier la liste de distance euclidienne : 
        listDistancesEuclidienne.sort(key=sort_item, reverse= False)
        print(listDistancesEuclidienne)

        #Récupérer la liste des images :
        listImage = []
        for i in listDistancesEuclidienne:
            listImage.append(i[0])
    
        #Récupérer la valeur de la signature : 
        listeSignatureEuclidinne = []
        for i in listDistancesEuclidienne:
            listeSignatureEuclidinne.append(i[2])
        return render(request, "search/search.html",{'form':form, 'msg':msg,'list':listImage,'listeSignatureEuclidinne':listeSignatureEuclidinne})
    else : 
        form=UploadFileForm()
        return render(request, "search/search.html",{'form':form})
   
def showHome(request):
    return render(request, 'search/home.html', {})