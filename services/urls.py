from django.urls import path
from . import views


urlpatterns = [
    path('auth/verify', views.verify_token),
    path('auth/register', views.register),
    path('auth/send-otp', views.send_otp),
    path('auth/verify-otp', views.verify_otp),
    path('auth/login', views.login_email_password),
    path('auth/forgot-password', views.forgot_password),
    path('me', views.me),
    path('pets', views.pets_list_create),
    path('pets/<str:pet_id>', views.pets_detail),
    path('scans', views.scans_create),
    path('history', views.history_list),
    path('admin/analytics', views.admin_analytics),
]


