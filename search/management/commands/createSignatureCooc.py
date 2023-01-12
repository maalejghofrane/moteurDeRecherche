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
            nom,sig,corr=cooccurence(dataSet[i])
            correloImages.append(corr)
            for i in range(0, 256, 1) : 
                sig[i] = corr[i][i]
            signatureImages.append(sig) 
            listeDesNom.append(nom)
            signatureEtNom.append((nom,sig))
        print(signatureEtNom[0][1])  
        arr = signatureEtNom

        with open(os.getcwd()+"/static/signatures/signaturesCooccurrence", "wb") as f:
            pickle.dump(arr, f)
        name = options['name']        
        self.stdout.write(self.style.SUCCESS('SUCCESS %s!' % name))
