import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

SECRET_KEY = 'dev-secret-key'
DEBUG = False

ALLOWED_HOSTS = [
    'testyourskin.online',
    'www.testyourskin.online',
    '91.99.119.240',
    '127.0.0.1',
]


INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'siteapp',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [
            str(BASE_DIR),  # D:\Project_frontend -> templates are HTML files at root
        ],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'project.wsgi.application'

# Database: fallback to SQLite if Postgres env not provided
if os.environ.get('POSTGRES_HOST'):
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.postgresql',
            'NAME': os.environ.get('POSTGRES_DB', 'project_db'),
            'USER': os.environ.get('POSTGRES_USER', 'postgres'),
            'PASSWORD': os.environ.get('POSTGRES_PASSWORD', ''),
            'HOST': os.environ.get('POSTGRES_HOST', '127.0.0.1'),
            'PORT': os.environ.get('POSTGRES_PORT', '5432'),
        }
    }
else:
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': str(BASE_DIR / 'backend' / 'db.sqlite3'),
        }
    }

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator', 'OPTIONS': {'min_length': 8}},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
    {'NAME': 'siteapp.validators.ComplexityValidator'},
]

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

STATIC_URL = '/static/'
STATICFILES_DIRS = [
    str(BASE_DIR / 'backend' / 'static'),
]
STATIC_ROOT = str(BASE_DIR / 'backend' / 'staticfiles')

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Auth redirects
LOGIN_URL = '/login/'
LOGIN_REDIRECT_URL = '/upload/'
LOGOUT_REDIRECT_URL = '/'

# Email: defaults to console backend; can be overridden via environment
import os as _os
EMAIL_BACKEND = _os.environ.get('EMAIL_BACKEND', 'django.core.mail.backends.console.EmailBackend')
EMAIL_HOST = _os.environ.get('EMAIL_HOST', 'smtp.gmail.com')
EMAIL_PORT = int(_os.environ.get('EMAIL_PORT', '587'))
EMAIL_USE_TLS = _os.environ.get('EMAIL_USE_TLS', '1') == '1'
EMAIL_USE_SSL = _os.environ.get('EMAIL_USE_SSL', '0') == '1'
EMAIL_HOST_USER = _os.environ.get('EMAIL_HOST_USER', '')
EMAIL_HOST_PASSWORD = _os.environ.get('EMAIL_HOST_PASSWORD', '')
DEFAULT_FROM_EMAIL = _os.environ.get('DEFAULT_FROM_EMAIL', 'no-reply@teim.co.in')

# Media files (uploads)
MEDIA_URL = '/media/'
MEDIA_ROOT = str(BASE_DIR / 'backend' / 'media')

# Sessions: expire after inactivity
SESSION_COOKIE_AGE = 30 * 60  # 30 minutes
SESSION_SAVE_EVERY_REQUEST = False


