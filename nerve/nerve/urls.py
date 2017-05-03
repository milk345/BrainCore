"""nerve URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.10/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.contrib import admin
from brain import views as brain_views  # new

urlpatterns = [
    url(r'^$', brain_views.index),  # new
    url(r'^admin/', admin.site.urls),
    url(r'^test/', brain_views.test),
    url(r'^create/$', brain_views.create, name='create'),
    url(r'^pratice/$', brain_views.pratice, name='pratice'),
    url(r'^predict/$', brain_views.predict, name='predict'),
    url(r'^soundTest/$', brain_views.upload_file, name='upload_file'),
]
