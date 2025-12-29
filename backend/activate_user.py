#!/usr/bin/env python
"""
Script to activate a user account (for development/testing)
Usage: python activate_user.py <username_or_email>
"""
import os
import sys
import django

# Setup Django
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')
django.setup()

from django.contrib.auth.models import User

def activate_user(username_or_email):
    """Activate a user account by username or email"""
    try:
        if '@' in username_or_email:
            user = User.objects.get(email__iexact=username_or_email)
        else:
            user = User.objects.get(username__iexact=username_or_email)
        
        if user.is_active:
            print(f"User '{user.username}' is already active.")
            return True
        
        user.is_active = True
        user.save(update_fields=['is_active'])
        print(f"[SUCCESS] Successfully activated user: {user.username} (Email: {user.email})")
        return True
    except User.DoesNotExist:
        print(f"[ERROR] User not found: {username_or_email}")
        return False
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        return False

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python activate_user.py <username_or_email>")
        print("\nExample:")
        print("  python activate_user.py Ruchita10")
        print("  python activate_user.py user@example.com")
        sys.exit(1)
    
    username_or_email = sys.argv[1]
    activate_user(username_or_email)

