from django.db import models

# Create your models here.
class Video(models.Model):
    video=models.FileField(upload_to='video/%y')