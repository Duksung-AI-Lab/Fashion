from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from . import views

app_name = 'upload_image'

urlpatterns = [
    path('', views.index, name='upload_index'),
    path('result', views.post_clothes, name='post_clothes'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
