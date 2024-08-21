
from django.contrib import admin
from django.urls import path
from django.urls import include
from youtube_site import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('youtube_site.urls')),
    
]
