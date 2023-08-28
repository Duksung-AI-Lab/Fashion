from django.urls import path
from . import views


app_name = 'select_type'

urlpatterns = [
    path('', views.index, name='select_index'),
    path('result/shirts', views.post_shirts, name='post_shirts'),
    path('result/tshirts', views.post_tshirts, name='post_tshirts'),
    path('result/jeans', views.post_jeans, name='post_jeans'),
    path('test', views.get_ajax, name='ajax'),
]
