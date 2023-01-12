from django.core.management.base import BaseCommand
import os
import glob
import cv2 as cv 
from skimage import io
import numpy as np
import pickle


class Command(BaseCommand):
    help = 'Greet the user'
    
    def add_arguments(self, parser):
        parser.add_argument('name', type=str, help='Name of the user to greet')

    def handle(self, *args, **options):
        def correlogramme (emplacement): 
            sig =[0 for _ in range(256)]
            emplacement = str(emplacement)
            nom = emplacement[36:]
            image = io.imread(emplacement) 
            imageGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            corI1 = [[1 for _ in range(256)] for _ in range(256)]
            sig1 =[0 for _ in range(256)]
            distance1=0
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

        def readPictureDataSet () : 
            images = [file for file in glob.glob(FOLDERDATASET)]
            print(images)
            return images
        FOLDERDATASET = os.getcwd()+"\static\dataset\*"
        dataSet = readPictureDataSet ()
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
        arr = signatureEtNom

        with open(os.getcwd()+"/static/signatures/signaturesCorrelogramme", "wb") as f:
            pickle.dump(arr, f) 
        name = options['name']        
        self.stdout.write(self.style.SUCCESS('SUCCESS %s!' % name))
