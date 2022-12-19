from django.db import models

class Predict(models.Model):
    name = models.CharField(max_length=100, null=True, blank=True)
    file = models.FileField()

    def __str__(self):             
        return self.name
