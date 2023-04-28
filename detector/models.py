from django.db import models

# Create your models here.
class Video(models.Model):
    video=models.FileField(upload_to='video/%y')

class TimeStamp(models.Model):
    start=models.CharField(max_length=9)
    duration=models.CharField(max_length=9)