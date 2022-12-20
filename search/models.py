from django.db import models
from django.core.files.storage import FileSystemStorage
import os 

class Predict(models.Model):
    fs = FileSystemStorage(location=os.getcwd()+'/static/requete')
    name = models.CharField(max_length=100, null=True, blank=True)
    file = models.FileField(storage=fs)

    def __str__(self):             
        return self.name
