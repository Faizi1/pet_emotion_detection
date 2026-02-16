from django.urls import path

from . import views


urlpatterns = [
    path("verify-receipt", views.verify_receipt),
    path("status", views.subscription_status),
    path("restore", views.restore_purchases),
    path("webhook", views.app_store_webhook),
    path("admin/list", views.admin_list_subscriptions),
    path("plans", views.list_plans),
]
