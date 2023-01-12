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
        with open(os.getcwd()+"/static/signatures/signaturesCooccurrence", "rb") as f:
            coocc = pickle.load(f,encoding="latin1")
            print(coocc)
        with open(os.getcwd()+"/static/signatures/signaturesCorrelogramme", "rb") as f:
            corr = pickle.load(f,encoding="latin1")
            print(corr)

        combined_features = np.concatenate((coocc[0][1], corr[0][1]))
        arr = combined_features 
        with open(os.getcwd()+"/static/signatures/signaturesCorrelogrammeEtHistogramme", "wb") as f:
            pickle.dump(arr, f)
        
        name = options['name']        
        self.stdout.write(self.style.SUCCESS('SUCCESS %s!' % name))
