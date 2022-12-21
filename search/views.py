from django.shortcuts import render
from .models import Predict
from .forms import UploadFileForm
import pathlib
import os
from skimage import io
import cv2 as cv 
from scipy.spatial import distance

# Lire Image 1 : 
img1 = io.imread(os.getcwd()+'/static/dataset/image1.png') 
img1Gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
print(img1Gray)
corI1 = [[1 for _ in range(256)] for _ in range(256)]
sig1 =[0 for _ in range(256)]
distance1=0

# Calcule corrélogramme de l'image 1 avec distance d=1 : 
for x in range(0, 239, 1) : 
  for y in range(0, 239, 1) : 
      o = img1Gray[x][y]
      o1=img1Gray[x-1][y-1]
      corI1[o][o1]=corI1[o][o1]+1
      o2=img1Gray[x-1][y]
      corI1[o][o2]=corI1[o][o2]+1
      o3=img1Gray[x-1][y+1]
      corI1[o][o3]=corI1[o][o3]+1
      o4=img1Gray[x][y-1]
      corI1[o][o4]=corI1[o][o4]+1
      o6=img1Gray[x][y+1]
      corI1[o][o6]=corI1[o][o6]+1
      o7=img1Gray[x+1][y-1]
      corI1[o][o7]=corI1[o][o7]+1
      o8=img1Gray[x+1][y]
      corI1[o][o8]=corI1[o][o8]+1
      o9=img1Gray[x+1][y+1]
      corI1[o][o9]=corI1[o][o9]+1
      
# Lire Image 2 : 
img2 = io.imread(os.getcwd()+'/static/dataset/image2.png') 
image2Gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
print(image2Gray)
corI2 = [[1 for _ in range(256)] for _ in range(256)]
sig2 =[0 for _ in range(256)]
distance2=0

#Calcule corrélogramme de l'image 2 avec distance d=1 : 
for x in range(0, 239, 1) : 
  for y in range(0, 239, 1) : 
      o = image2Gray[x][y]
      o1=image2Gray[x-1][y-1]
      corI2[o][o1]=corI2[o][o1]+1
      o2=image2Gray[x-1][y]
      corI2[o][o2]=corI2[o][o2]+1
      o3=image2Gray[x-1][y+1]
      corI2[o][o3]=corI2[o][o3]+1
      o4=image2Gray[x][y-1]
      corI2[o][o4]=corI2[o][o4]+1
      o6=image2Gray[x][y+1]
      corI2[o][o6]=corI2[o][o6]+1
      o7=image2Gray[x+1][y-1]
      corI2[o][o7]=corI2[o][o7]+1
      o8=image2Gray[x+1][y]
      corI2[o][o8]=corI2[o][o8]+1
      o9=image2Gray[x+1][y+1]
      corI2[o][o9]=corI2[o][o9]+1

#Lire Image 3 :
img3 = io.imread(os.getcwd()+'/static/dataset/image3.png') 
image3Gray = cv.cvtColor(img3, cv.COLOR_BGR2GRAY)
print(image3Gray)
corI3 = [[1 for _ in range(256)] for _ in range(256)]
sig3 =[0 for _ in range(256)]
distance3=0

#Calcule corrélogramme de l'image 3 avec distance d=1 : 
for x in range(0, 239, 1) : 
  for y in range(0, 239, 1) : 
      o = image3Gray[x][y]
      o1=image3Gray[x-1][y-1]
      corI3[o][o1]=corI3[o][o1]+1
      o2=image3Gray[x-1][y]
      corI3[o][o2]=corI3[o][o2]+1
      o3=image3Gray[x-1][y+1]
      corI3[o][o3]=corI3[o][o3]+1
      o4=image3Gray[x][y-1]
      corI3[o][o4]=corI3[o][o4]+1
      o6=image3Gray[x][y+1]
      corI3[o][o6]=corI3[o][o6]+1
      o7=image3Gray[x+1][y-1]
      corI3[o][o7]=corI3[o][o7]+1
      o8=image3Gray[x+1][y]
      corI3[o][o8]=corI3[o][o8]+1
      o9=image3Gray[x+1][y+1]
      corI3[o][o9]=corI3[o][o9]+1

#Lire Image Requete : 
img10 = io.imread(os.getcwd()+'/static/dataset/image1.png') 
img10Gray = cv.cvtColor(img10, cv.COLOR_BGR2GRAY)
print(img10Gray)
corI10 = [[1 for _ in range(256)] for _ in range(256)]
sigreq =[0 for _ in range(256)]
distance10=0

# Calcule corrélogramme de l'image 10 : 
for x in range(0, 239, 1) : 
  for y in range(0, 239, 1) : 
      o = img10Gray[x][y]
      o1=img10Gray[x-1][y-1]
      corI10[o][o1]=corI10[o][o1]+1
      o2=img10Gray[x-1][y]
      corI10[o][o2]=corI10[o][o2]+1
      o3=img10Gray[x-1][y+1]
      corI10[o][o3]=corI10[o][o3]+1
      o4=img10Gray[x][y-1]
      corI10[o][o4]=corI10[o][o4]+1
      o6=img10Gray[x][y+1]
      corI10[o][o6]=corI10[o][o6]+1
      o7=img10Gray[x+1][y-1]
      corI10[o][o7]=corI10[o][o7]+1
      o8=img10Gray[x+1][y]
      corI10[o][o8]=corI10[o][o8]+1
      o9=img10Gray[x+1][y+1]
      corI10[o][o9]=corI10[o][o9]+1

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
  return item[1]

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
        return render(request, "search/search.html",{'form':form, 'msg':msg,'list':listImage,'listeSignatureEuclidinne':listeSignatureEuclidinne})
    else : 
        form=UploadFileForm()
        return render(request, "search/search.html",{'form':form})
   
def showHome(request):
    return render(request, 'search/home.html', {})