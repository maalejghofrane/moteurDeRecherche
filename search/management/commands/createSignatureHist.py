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
        def histogramme (emplacement) :
            sig=[] 
            emplacement = str(emplacement)
            nom = emplacement[36:]
            img = cv.imread(emplacement,0)
            h = cv.calcHist([img], [0], None, [256], [0, 255])
            return nom,h 
        def readPictureDataSet () : 
            images = [file for file in glob.glob(FOLDERDATASET)]
            print(images)
            return images
        FOLDERDATASET = os.getcwd()+"\static\dataset\*"
        dataSet = readPictureDataSet ()
        histogrammes=[]
        listeDesNom = []
        histEtNom = []
        for i in range(len(dataSet)):
            nom,hist=histogramme(dataSet[i])
            histogrammes.append(hist)
            listeDesNom.append(nom)
            histEtNom.append((nom,hist))
        # print(histEtNom[0][1]) 
        # print(histEtNom[0]) 
        arr = histEtNom

        with open(os.getcwd()+"/static/signatures/signaturesHistogramme", "wb") as f:
            pickle.dump(arr, f) 
        name = options['name']        
        self.stdout.write(self.style.SUCCESS('SUCCESS %s!' % name))
