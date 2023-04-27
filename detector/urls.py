from django.contrib import admin
from django.urls import path,include
from detector import views

urlpatterns = [
    path('',views.home,name='home'),
    path('upload/',views.save_video,name='upload'),
    path('timestamp/',views.extract_snippet,name='timestamp')
]