from django.urls import path
from . import views


urlpatterns = [
    path('auth/verify', views.verify_token),
    path('auth/register', views.register),
    path('auth/verify-otp-and-register', views.verify_otp_and_register),
    path('auth/send-otp', views.send_otp),
    path('auth/verify-otp', views.verify_otp),
    path('auth/google-signin', views.google_signin),
    path('auth/apple-signin', views.apple_signin),
    path('auth/login', views.login_email_password),
    path('auth/forgot-password', views.forgot_password),
    path('auth/change-password', views.change_password),
    path('me', views.me),
    path('profile', views.update_profile),
    path('pets', views.pets_list_create),
    path('pets/<str:pet_id>', views.pets_detail),
    path('scans', views.scans_create),
    path('history', views.history_list),
    path('admin/analytics', views.admin_analytics),
    
    # Community/Posts endpoints
    path('community/posts', views.community_posts_list),
    path('community/posts/my', views.my_posts_list),
    path('community/posts/create', views.community_post_create),
    path('community/posts/<str:post_id>', views.community_post_detail),
    path('community/posts/<str:post_id>/delete', views.community_post_delete),
    path('community/posts/<str:post_id>/like', views.toggle_post_like),
    path('community/posts/<str:post_id>/share', views.share_post),
    path('community/posts/<str:post_id>/comments', views.post_comments_list),
    path('community/comments/create', views.create_comment),
    path('community/comments/<str:post_id>/<str:comment_id>/delete', views.delete_comment),
    
    # Utility endpoints
    path('admin/storage-config', views.check_storage_config),
    path('admin/sms-config', views.check_sms_config),
    path('sms/message-status/<str:message_sid>', views.sms_message_status),
    path('sms/verify-phone', views.verify_phone_number),
    path('sms/account-status', views.twilio_account_status),
    
    # Support/Help Desk endpoints (Simplified)
    path('support/send', views.send_support_message),
    path('support/my', views.my_support_messages),
    
    # (Removed admin support management endpoints added for new dashboard)
    
    # (Removed new dashboard/admin management and debug endpoints)
]
