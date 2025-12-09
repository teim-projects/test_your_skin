#!/usr/bin/env python
"""
Script to check user account status
"""
import os
import sys
import django

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')
django.setup()

from django.contrib.auth.models import User
from django.contrib.auth import authenticate

def check_user(username_or_email):
    """Check user account details"""
    try:
        if '@' in username_or_email:
            user = User.objects.get(email__iexact=username_or_email)
        else:
            user = User.objects.get(username__iexact=username_or_email)
        
        print(f"User found: {user.username}")
        print(f"  Email: {user.email}")
        print(f"  Is Active: {user.is_active}")
        print(f"  Is Staff: {user.is_staff}")
        print(f"  Is Superuser: {user.is_superuser}")
        print(f"  Date Joined: {user.date_joined}")
        print(f"  Last Login: {user.last_login}")
        return user
    except User.DoesNotExist:
        print(f"[ERROR] User not found: {username_or_email}")
        return None

if __name__ == '__main__':
    if len(sys.argv) < 2:
        username = 'Ruchita10'
    else:
        username = sys.argv[1]
    check_user(username)






