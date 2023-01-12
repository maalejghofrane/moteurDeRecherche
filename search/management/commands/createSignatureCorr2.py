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
        def correlogramme2(emplacement): 
            sig =[0 for _ in range(256)]
            emplacement = str(emplacement)
            nom = emplacement[36:]
            image = io.imread(emplacement) 
            gray_image  = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            # Create an empty correlogram
            correlogram = np.zeros((256, 256))
            # Iterate over the rows and columns of the image
            for i in range(gray_image.shape[0]):
                for j in range(gray_image.shape[1]):
                    x = gray_image[i, j]
                # Iterate over the surrounding pixels
                    for k in range(-1, 2):
                        for l in range(-1, 2):
                            if i + k >= 0 and i + k < gray_image.shape[0] and j + l >= 0 and j + l < gray_image.shape[1]:
                                y = gray_image[i + k, j + l]
                                correlogram[x, y] += 1
            # Normalize the correlogram
            correlogram = correlogram / np.sum(correlogram)
            return nom,sig,correlogram 
        
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
            nom,sig,corr=correlogramme2(dataSet[i])
            correloImages.append(corr)
            for i in range(0, 256, 1) : 
                sig[i] = corr[i][i]
            signatureImages.append(sig) 
            listeDesNom.append(nom)
            signatureEtNom.append((nom,sig))
        print(signatureEtNom[0][1])  
        arr = signatureEtNom

        with open(os.getcwd()+"/static/signatures/signaturesCorrelogramme2", "wb") as f:
            pickle.dump(arr, f) 
        name = options['name']        
        self.stdout.write(self.style.SUCCESS('SUCCESS %s!' % name))
