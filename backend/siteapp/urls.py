from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', views.index, name='index'),
    path('about/', views.about, name='about'),
    path('login/', views.login_page, name='login'),
    path('register/', views.register_page, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('user/', views.user_profile, name='user'),
    # Password reset
    path('password-reset/', auth_views.PasswordResetView.as_view(template_name='password_reset.html'), name='password_reset'),
    path('password-reset/done/', auth_views.PasswordResetDoneView.as_view(template_name='password_reset_done.html'), name='password_reset_done'),
    path('reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(template_name='password_reset_confirm.html'), name='password_reset_confirm'),
    path('reset/done/', auth_views.PasswordResetCompleteView.as_view(template_name='password_reset_complete.html'), name='password_reset_complete'),
    path('doctors/', views.doctors, name='doctors'),
    path('treatment/', views.treatment, name='treatment'),
    path('upload/', views.image_upload, name='image_upload'),
    path('gallery/', views.gallery, name='gallery'),
    path('contact/', views.contact, name='contact'),
    path('download/', views.download_page, name='download'),
    path('predict/', views.predict, name='predict'),
    path('clear-model-cache/', views.clear_model_cache, name='clear_model_cache'),

    path('user/password/', views.change_password, name='change_password'),
    path('send-email-otp/', views.send_email_otp, name='send_email_otp'),
    path('verify-email-otp/', views.verify_email_otp, name='verify_email_otp'),
    path('start-diagnostics/', views.start_diagnostics, name='start_diagnostics'),
]


