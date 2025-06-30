from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_image, name='upload_image'),
    path('process/', views.process_image_and_get_neighbors, name='process_image'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
