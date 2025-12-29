import os
import sys
import django
from django.conf import settings

# Setup Django
sys.path.append('.')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')
django.setup()

from django.core.mail import send_mail

# Test email sending
try:
    send_mail(
        'Test Subject',
        'Test message body',
        'no-reply@teim.co.in',
        ['test@example.com'],
        fail_silently=False
    )
    print("Email sent successfully (to console)")
except Exception as e:
    print(f"Email sending failed: {e}")

print(f"Email backend: {settings.EMAIL_BACKEND}")
print(f"Debug mode: {settings.DEBUG}")