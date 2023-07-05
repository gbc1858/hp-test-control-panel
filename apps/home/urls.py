from django.urls import path, re_path
from apps.home import views
#
# from django.conf.urls import url

urlpatterns = [
    path('api/set-address', views.set_address, name='set-address'),
    path('api/run-task', views.run_task, name='run-task'),
    path('api/get-last-task', views.get_last_task, name='get-last-task'),
    path('api/get-attn', views.get_attn, name='get-attn'),
    path('api/get-data-to-process', views.get_data_to_process, name='get-data-to-process'),
    path('api/get-data-to-convolve', views.get_data_to_convolve, name='get-data-to-convolve'),

    # The home page
    path('', views.index, name='home'),
    # Matches any html file
    re_path(r'^.*\.*', views.pages, name='pages'),
    # path("", views.home_view, name="")
]
