import os
import uuid
from django.db import models
from django.contrib.auth.models import User


def upload_to(instance, filename):
    ext = os.path.splitext(filename)[1].lower()
    return os.path.join('uploads', f"{uuid.uuid4().hex}{ext}")


class Analysis(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    image = models.ImageField(upload_to=upload_to, null=True, blank=True)
    original_filename = models.CharField(max_length=255, blank=True)
    size_bytes = models.BigIntegerField(default=0)
    content_type = models.CharField(max_length=64, blank=True)
    symptoms = models.JSONField(default=list)
    results = models.JSONField(default=dict)
    quality_info = models.JSONField(default=dict)

    def __str__(self):
        return f"Analysis {self.id}"


class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    full_name = models.CharField(max_length=255, blank=True)
    age = models.PositiveIntegerField(null=True, blank=True)
    gender = models.CharField(max_length=32, blank=True)
    city = models.CharField(max_length=128, blank=True)
    mobile = models.CharField(max_length=20, blank=True)
    location = models.CharField(max_length=255, blank=True)
    bio = models.TextField(blank=True)
    avatar = models.ImageField(upload_to=upload_to, null=True, blank=True)

    def __str__(self):
        return f"Profile of {self.user.username}"

