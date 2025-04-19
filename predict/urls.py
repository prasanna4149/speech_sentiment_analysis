from django.urls import path
from .views import predict_audio

urlpatterns = [
    path('predict/', predict_audio, name='predict_audio'),
]
