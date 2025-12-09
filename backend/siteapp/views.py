import json
import os
import random
import time
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')  # prefer tf.keras legacy loader
from typing import List, Tuple
from django.http import JsonResponse, HttpRequest, HttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth import update_session_auth_hash
from django.conf import settings
from django.core.exceptions import ValidationError
from django.contrib.auth.password_validation import validate_password
from django.core import signing
from django.core.mail import send_mail
from django.views.decorators.http import require_POST
from PIL import Image, ImageStat
from io import BytesIO
import numpy as np
import logging
import h5py
import sys
import uuid
import imghdr
from .models import Analysis, Profile

from .ml.disease_prediction_service import DiseasePredictionService

# Initialize logger first
logger = logging.getLogger(__name__)

# TensorFlow import with detailed error handling
try:
    import tensorflow as tf
    from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
    logger.info(f"TensorFlow {tf.__version__} imported successfully")
except ImportError as e:
    logger.error(f"TensorFlow import failed: {e}")
    tf = None
    densenet_preprocess = None
except Exception as e:
    logger.error(f"Unexpected error importing TensorFlow: {e}")
    tf = None
    densenet_preprocess = None

# Lazy-loaded global prediction service to avoid reloading per request
_PREDICTION_SERVICE = None

def _get_prediction_service():
    """Get or create the prediction service instance"""
    global _PREDICTION_SERVICE
    if _PREDICTION_SERVICE is None:
        try:
            # Try to load existing models first (prioritize the new ResNet-based hybrid model)
            model_paths = [
                os.path.join(str(settings.BASE_DIR), 'hybrid_best.h5'),
                os.path.join(str(settings.BASE_DIR), 'my_model.h5'),
            ]
            
            model_loaded = False
            for model_path in model_paths:
                if os.path.exists(model_path):
                    logger.info(f"Attempting to load model: {model_path}")
                    _PREDICTION_SERVICE = DiseasePredictionService(
                        dataset_path=os.path.join(str(settings.BASE_DIR), 'Dataset'),
                        model_path=model_path
                    )
                    if _PREDICTION_SERVICE.model is not None:
                        model_loaded = True
                        logger.info(f"Successfully loaded model from {model_path}")
                        break
            
            if not model_loaded:
                # Create service without model - will use fallback
                _PREDICTION_SERVICE = DiseasePredictionService(
                    dataset_path=os.path.join(str(settings.BASE_DIR), 'Dataset')
                )
                logger.warning("No compatible model found, using fallback service")
                
        except Exception as e:
            logger.error(f"Error initializing prediction service: {e}")
            _PREDICTION_SERVICE = DiseasePredictionService(
                dataset_path=os.path.join(str(settings.BASE_DIR), 'Dataset')
            )
    
    return _PREDICTION_SERVICE

def _clear_model_cache():
    """Clear the cached prediction service to force reload"""
    global _PREDICTION_SERVICE
    _PREDICTION_SERVICE = None
    logger.info("Prediction service cache cleared - will reload on next request")

def clear_model_cache(request):
    """Debug endpoint to clear model cache"""
    _clear_model_cache()
    return JsonResponse({'status': 'success', 'message': 'Prediction service cache cleared'})

# Legacy function for backward compatibility - now uses prediction service
def _get_model():
    """Legacy function - now returns the model from prediction service"""
    service = _get_prediction_service()
    return service.model if service else None

# Legacy functions for backward compatibility - now use prediction service
def _get_class_names(num_classes: int) -> List[str]:
    """Legacy function - now gets class names from prediction service"""
    service = _get_prediction_service()
    return service.class_names if service else [f'Class {i}' for i in range(num_classes)]

def _prepare_image_for_densenet(img: Image.Image) -> np.ndarray:
    """Legacy function - now uses prediction service preprocessing"""
    service = _get_prediction_service()
    if service:
        return service.preprocess_image(img)
    else:
        # Fallback preprocessing
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((224, 224))
        arr = np.array(img, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)
        if densenet_preprocess is not None:
            arr = densenet_preprocess(arr)
        return arr

def _topk_from_logits(logits: np.ndarray, k: int = 3) -> Tuple[List[int], List[float]]:
    """Legacy function - converts logits to top-k predictions"""
    probs = tf.nn.softmax(logits, axis=-1).numpy().reshape(-1)
    indices = probs.argsort()[::-1][:k]
    confidences = probs[indices].tolist()
    return indices.tolist(), confidences


def index(request: HttpRequest) -> HttpResponse:
    return render(request, 'index.html')


def about(request: HttpRequest) -> HttpResponse:
    return render(request, 'About Us.html')


def login_page(request: HttpRequest) -> HttpResponse:
    if request.method == 'POST':
        username_or_email = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')
        # Simple lockout after multiple failed attempts (per session)
        fail_key = 'login_fail_count'
        lock_key = 'login_locked_until'
        from datetime import datetime, timedelta
        now_ts = datetime.utcnow().timestamp()
        locked_until = request.session.get(lock_key, 0)
        if now_ts < locked_until:
            return render(request, 'Login.html', { 'error': 'Too many failed attempts. Please try again later or reset your password.' })

        if not username_or_email or not password:
            return render(request, 'Login.html', { 'error': 'Please enter both username/email and password.', 'username_value': username_or_email })

        # Determine if input is email or username, but never reveal which is wrong
        username = username_or_email
        user_obj = None
        
        # Check if input is an email (case-insensitive lookup)
        if '@' in username_or_email:
            try:
                user_obj = User.objects.get(email__iexact=username_or_email)
                username = user_obj.username
            except User.DoesNotExist:
                username = username_or_email  # continue with authenticate for generic error
        else:
            # Try to find user by username (case-insensitive)
            try:
                user_obj = User.objects.get(username__iexact=username_or_email)
                username = user_obj.username
            except User.DoesNotExist:
                pass  # Will try authenticate with original input
        
        # User verification is handled via OTP during registration, no additional checks needed
        
        # Authenticate user (Django's authenticate only works with active users by default)
        # In DEBUG mode, we'll manually check password for inactive users
        user = authenticate(request, username=username, password=password)
        
        # If authentication failed but user exists, try manual password check (for inactive users in DEBUG mode)
        if not user and user_obj:
            if settings.DEBUG and user_obj.check_password(password):
                user = user_obj
                # Auto-activate user in development mode
                if not user.is_active:
                    user.is_active = True
                    user.save(update_fields=['is_active'])
        
        if user:
            login(request, user)
            request.session[fail_key] = 0
            return redirect('image_upload')
        
        # Increment failure counter
        fails = int(request.session.get(fail_key, 0)) + 1
        request.session[fail_key] = fails
        if fails >= 5:
            request.session[lock_key] = now_ts + 15 * 60  # 15 minutes lockout
        
        return render(request, 'Login.html', { 
            'error': 'Login failed. Please check your credentials and try again.', 
            'username_value': username_or_email 
        })
    return render(request, 'Login.html')


@require_POST
def send_email_otp(request: HttpRequest) -> JsonResponse:
    try:
        payload = json.loads(request.body.decode('utf-8')) if request.body else {}
    except json.JSONDecodeError:
        payload = {}
    email = str(payload.get('email', '')).strip()
    if not email:
        return JsonResponse({'ok': False, 'error': 'Email is required.'}, status=400)
    try:
        from django.core.validators import validate_email
        validate_email(email)
    except ValidationError:
        return JsonResponse({'ok': False, 'error': 'Enter a valid email address.'}, status=400)

    if User.objects.filter(email__iexact=email).exists():
        return JsonResponse({'ok': False, 'error': 'Email already registered.'}, status=400)

    session_info = request.session.get('email_otp') or {}
    now = int(time.time())
    cooldown = 60
    sent_at = int(session_info.get('sent_at', 0))
    if session_info.get('email') == email.lower() and sent_at and now - sent_at < cooldown:
        wait_for = cooldown - (now - sent_at)
        return JsonResponse({
            'ok': False,
            'error': f'Please wait {wait_for} seconds before requesting a new OTP.',
            'resend_in': wait_for
        }, status=429)

    otp_code = f"{random.randint(0, 999999):06d}"
    expires_at = now + 600  # 10 minutes
    otp_payload = {
        'email': email.lower(),
        'otp': otp_code,
        'sent_at': now,
        'expires_at': expires_at,
    }
    subject = 'Verify your email address'
    message = (
        f'Your verification code is {otp_code}.\n\n'
        'It expires in 10 minutes. If you did not initiate this request, you can ignore this message.'
    )
    from_email = getattr(settings, 'DEFAULT_FROM_EMAIL', 'no-reply@teim.co.in')
    try:
        send_mail(subject, message, from_email, [email], fail_silently=False)
    except Exception as exc:
        logger.warning("Failed to send email OTP: %s", exc)
        request.session['email_otp'] = otp_payload
        request.session.pop('email_verified_email', None)
        request.session.modified = True
        if getattr(settings, 'DEBUG', False):
            return JsonResponse({
                'ok': True,
                'message': f'OTP generated (development mode). Use code: {otp_code}',
                'resend_in': cooldown
            })
        return JsonResponse({'ok': False, 'error': 'Could not send OTP right now. Please try again later.'}, status=500)

    request.session['email_otp'] = otp_payload
    request.session.pop('email_verified_email', None)
    request.session.modified = True
    return JsonResponse({'ok': True, 'message': 'OTP sent to your email address.', 'resend_in': cooldown})


@require_POST
def verify_email_otp(request: HttpRequest) -> JsonResponse:
    try:
        payload = json.loads(request.body.decode('utf-8')) if request.body else {}
    except json.JSONDecodeError:
        payload = {}
    email = str(payload.get('email', '')).strip().lower()
    otp = str(payload.get('otp', '')).strip()
    if not email or not otp:
        return JsonResponse({'ok': False, 'error': 'Email and OTP are required.'}, status=400)

    entry = request.session.get('email_otp') or {}
    if entry.get('email') != email:
        return JsonResponse({'ok': False, 'error': 'OTP not found. Please request a new code.'}, status=400)

    now = int(time.time())
    expires_at = int(entry.get('expires_at', 0))
    if expires_at and now > expires_at:
        request.session.pop('email_otp', None)
        request.session.modified = True
        return JsonResponse({'ok': False, 'error': 'OTP has expired. Please request a new code.'}, status=400)

    if entry.get('otp') != otp:
        return JsonResponse({'ok': False, 'error': 'Invalid OTP. Please try again.'}, status=400)

    request.session['email_verified_email'] = email
    request.session.pop('email_otp', None)
    request.session.modified = True
    return JsonResponse({'ok': True, 'message': 'Email verified successfully.'})


def register_page(request: HttpRequest) -> HttpResponse:
    if request.method == 'POST':
        full_name = request.POST.get('full_name')
        age = request.POST.get('age')
        gender = request.POST.get('gender')
        city = request.POST.get('city')
        mobile = request.POST.get('mobile')
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        confirm = request.POST.get('confirmpassword')
        # Username: 3-20 alphanumeric
        import re
        # Basic validation for added fields
        form_values = request.POST
        session_verified_email = str(request.session.get('email_verified_email') or '').strip().lower()
        email_norm = (email or '').strip().lower()

        def render_error(message: str) -> HttpResponse:
            verified = bool(session_verified_email) and session_verified_email == email_norm
            return render(request, 'Register.html', {
                'error': message,
                'form_values': form_values,
                'email_verified': verified,
                'verified_email': session_verified_email if verified else ''
            })
        if not full_name:
            return render_error('Please enter patient name.')
        try:
            age_int = int(age or '0')
            if age_int < 0 or age_int > 120:
                return render_error('Please enter a valid age (0-120).')
        except ValueError:
            return render_error('Please enter a valid age (number).')
        if not gender:
            return render_error('Please select gender.')
        if not city:
            return render_error('Please enter city.')
        if not re.fullmatch(r"[0-9]{10}", (mobile or '')):
            return render_error('Please enter a valid 10-digit mobile number.')
        if not username or not re.fullmatch(r'[A-Za-z0-9]{3,20}', username or ''):
            return render_error('Username must be 3-20 characters, letters and digits only.')
        # Email format
        from django.core.validators import validate_email
        try:
            validate_email(email)
        except ValidationError:
            return render_error('Please enter a valid email address.')
        # Unique username/email
        if User.objects.filter(username=username).exists():
            return render_error('Username already exists')
        if User.objects.filter(email=email).exists():
            return render_error('Email already registered')

        if session_verified_email != email_norm or not email_norm:
            return render_error('Please verify your email with the OTP before continuing.')
        # Password confirm and strength
        if not password or password != confirm:
            return render_error('Passwords do not match.')
        try:
            validate_password(password)
        except ValidationError as ve:
            return render_error(' '.join([str(m) for m in ve.messages]))

        user = User.objects.create_user(username=username, email=email, password=password)
        # User is active immediately (no email verification required)
        user.is_active = True
        # Optionally set first_name/last_name from full name (best-effort split)
        try:
            parts = (full_name or '').strip().split()
            if parts:
                user.first_name = parts[0]
                user.last_name = ' '.join(parts[1:])
                user.save(update_fields=['first_name', 'last_name'])
        except Exception:
            pass
        # Create and populate profile
        try:
            profile, _ = Profile.objects.get_or_create(user=user)
            profile.full_name = full_name or ''
            profile.age = age_int
            profile.gender = gender or ''
            profile.city = city or ''
            profile.mobile = mobile or ''
            profile.save()
        except Exception:
            pass
        # Prepare acknowledgment context
        ack = {
            'full_name': full_name,
            'age': age_int,
            'gender': gender,
            'city': city,
            'mobile': mobile,
            'email': email,
        }
        # Store for post-register auto-login
        request.session['post_register_user_id'] = user.id
        request.session.pop('email_verified_email', None)
        request.session.pop('email_otp', None)
        return render(request, 'register_success.html', { 'ack': ack })
    return render(request, 'Register.html', { 'email_verified': False, 'verified_email': '' })

def start_diagnostics(request: HttpRequest) -> HttpResponse:
    """Log in the newly registered user and redirect to image upload."""
    uid = request.session.get('post_register_user_id')
    try:
        if uid:
            user = User.objects.get(id=int(uid))
            login(request, user)
            # one-time use
            request.session.pop('post_register_user_id', None)
            return redirect('image_upload')
    except Exception:
        pass
    return redirect('login')






def doctors(request: HttpRequest) -> HttpResponse:
    return render(request, 'Doc.html')


def treatment(request: HttpRequest) -> HttpResponse:
    return render(request, 'Treatment.html')


@login_required
def image_upload(request: HttpRequest) -> HttpResponse:
    return render(request, 'image upload.html')


def logout_view(request: HttpRequest) -> HttpResponse:
    logout(request)
    return redirect('index')


def gallery(request: HttpRequest) -> HttpResponse:
    return render(request, 'gallery.html')


def contact(request: HttpRequest) -> HttpResponse:
    return render(request, 'contact.html')


def download_page(request: HttpRequest) -> HttpResponse:
    analysis_id = request.GET.get('id')
    analysis = None
    if analysis_id and str(analysis_id).isdigit():
        try:
            analysis = Analysis.objects.get(id=int(analysis_id))
        except Analysis.DoesNotExist:
            analysis = None
    context = {}
    if analysis:
        preds = analysis.results.get('predictions', []) if isinstance(analysis.results, dict) else []
        primary = preds[0] if preds else None
        context = {
            'analysis': analysis,
            'primary_disease': (primary or {}).get('disease', 'Unknown'),
            'primary_confidence': (primary or {}).get('confidence', ''),
            'symptoms_str': ', '.join(analysis.symptoms or []),
            'predictions': preds,
        }
    return render(request, 'download.html', context)


def predict(request: HttpRequest) -> JsonResponse:
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST allowed'}, status=405)

    image_file = request.FILES.get('image')
    symptoms_json = request.POST.get('symptoms')

    if not image_file:
        return JsonResponse({'error': 'No image provided'}, status=400)

    try:
        symptoms = json.loads(symptoms_json) if symptoms_json else []
    except json.JSONDecodeError:
        symptoms = []

    # Get prediction service
    try:
        service = _get_prediction_service()
        if not service:
            raise RuntimeError("Prediction service not available")
        # Check if model is loaded
        if service.model is None:
            model_path = os.path.join(str(settings.BASE_DIR), 'my_model.h5')
            raise RuntimeError(f"Model not loaded. Please ensure my_model.h5 exists at: {model_path}")
    except Exception as e:
        logger.error(f"Failed to get prediction service: {e}")
        return JsonResponse({
            'ok': False,
            'error': f'Prediction service unavailable: {str(e)}',
            'quality_ok': False,
            'predictions': [],
            'symptoms': symptoms,
            'filename': getattr(image_file, 'name', None),
        }, status=500)

    # Validate image quality and make prediction
    try:
        # Server-side validation
        MAX_SIZE = 1 * 1024 * 1024
        if getattr(image_file, 'size', 0) > MAX_SIZE:
            return JsonResponse({'ok': False, 'error': 'File too large. Maximum allowed size is 1 MB.'}, status=400)
        if not str(getattr(image_file, 'content_type', '')).startswith('image/'):
            return JsonResponse({'ok': False, 'error': 'Unsupported file type. Please upload an image file.'}, status=400)
        raw = image_file.read()
        buf = BytesIO(raw)
        try:
            img = Image.open(buf)
            img.verify()
        except Exception:
            return JsonResponse({'ok': False, 'error': 'Invalid image content.'}, status=400)
        buf.seek(0)
        img = Image.open(buf)
        
        # Validate image quality using the service
        quality_info = service.validate_image_quality(img)
        quality_ok = quality_info.get('quality_ok', False)
        
        # Make prediction using the service
        try:
            predictions_data = service.predict(img, top_k=2)
        except RuntimeError as e:
            logger.error(f"Prediction failed: {e}")
            return JsonResponse({
                'ok': False,
                'error': str(e),
                'quality_ok': quality_ok,
                'predictions': [],
                'symptoms': symptoms,
                'filename': getattr(image_file, 'name', None),
            }, status=500)
        except Exception as e:
            logger.exception(f"Unexpected error during prediction: {e}")
            return JsonResponse({
                'ok': False,
                'error': f'Prediction failed: {str(e)}',
                'quality_ok': quality_ok,
                'predictions': [],
                'symptoms': symptoms,
                'filename': getattr(image_file, 'name', None),
            }, status=500)

        # Format predictions for the frontend
        predictions = []
        for pred in predictions_data:
            predictions.append({
                'disease': pred['disease'],
                'confidence': pred['confidence'],
                'description': pred['description'],
                'symptoms': pred['symptoms'],
                'treatment': pred['treatment'],
                'urgency': pred['urgency'],
                'prevention': pred['prevention']
            })

        def _apply_symptom_weighting(preds, selected_symptoms):
            cleaned = {
                str(sym).strip().lower()
                for sym in (selected_symptoms or [])
                if sym and str(sym).strip().lower() not in {'none of the above', 'none'}
            }

            if not cleaned:
                for item in preds:
                    item.setdefault('original_confidence', item.get('confidence', 0.0))
                    item.setdefault('symptom_matches', [])
                    item.setdefault('symptom_match_ratio', 0.0)
                    item.setdefault('confidence_adjustment', 0.0)
                return preds

            BOOST = 0.25
            PENALTY = 0.05

            for item in preds:
                original_conf = float(item.get('confidence', 0.0))
                disease_symptoms = item.get('symptoms') or []
                disease_symptoms_lower = {
                    str(sym).strip().lower()
                    for sym in disease_symptoms if sym
                }
                match_lower = cleaned.intersection(disease_symptoms_lower)
                displayed_matches = [
                    sym for sym in disease_symptoms
                    if sym and sym.strip().lower() in match_lower
                ]
                user_ratio = (len(match_lower) / len(cleaned)) if cleaned else 0.0
                disease_ratio = (
                    len(match_lower) / len(disease_symptoms_lower)
                ) if disease_symptoms_lower else 0.0

                if match_lower:
                    boost = BOOST * (0.6 * user_ratio + 0.4 * disease_ratio)
                    penalty = 0.0
                else:
                    boost = 0.0
                    penalty = PENALTY

                adjusted = max(0.0, min(1.0, original_conf + boost - penalty))

                item['original_confidence'] = original_conf
                item['confidence'] = adjusted
                item['symptom_matches'] = displayed_matches
                item['symptom_match_ratio'] = user_ratio
                item['symptom_coverage_ratio'] = disease_ratio
                item['confidence_adjustment'] = adjusted - original_conf

            preds.sort(key=lambda x: x.get('confidence', 0.0), reverse=True)
            return preds

        predictions = _apply_symptom_weighting(predictions, symptoms)
        
        # Determine user-facing messaging for special cases
        user_message = None
        user_advice = None
        category = None

        # Case 1: Not a skin image or poor quality
        if not quality_ok or not quality_info.get('is_skin_like', True):
            user_message = 'The uploaded image is not recognized as a skin image or is of poor quality.'
            user_advice = 'Please upload a clear, focused image of the skin area requiring diagnosis.'
            category = 'Error'

        # If we have predictions, analyze confidences
        top_conf = predictions[0]['confidence'] if predictions else 0.0
        top_label = predictions[0]['disease'] if predictions else None

        # Case 2: Clear/Normal skin (dataset contains Clear_Skin)
        if top_label and top_label.lower().replace(' ', '_') in ['clear_skin', 'normal_skin'] and top_conf >= 0.60:
            user_message = 'No skin disease detected. The skin appears healthy and normal.'
            user_advice = 'Maintain good skincare habits: gentle cleansing, moisturize, and use sunscreen.'
            category = 'Clear / Healthy'

        # Case 3: Very low confidence - don't create "Other / Unknown", just keep original predictions
        # Filter will remove any "Other / Unknown" entries later
        LOW_CONF_THRESHOLD = 0.15
        if top_conf < LOW_CONF_THRESHOLD and (quality_ok and quality_info.get('is_skin_like', True)):
            # Keep original predictions but don't create "Other / Unknown"
            # If all predictions are filtered out later, frontend will handle empty results
            if user_message is None:
                user_message = 'The analyzed skin image has low confidence predictions. Please consult a dermatologist for further diagnosis.'
                user_advice = 'For accurate diagnosis, please consult a qualified dermatologist.'
                category = 'Low Confidence'

        # Persist analysis (randomized filename)
        saved_rel = None
        try:
            ext = '.jpg'
            try:
                fmt = imghdr.what(None, h=raw)
                if fmt:
                    ext = '.' + ('jpg' if fmt == 'jpeg' else fmt)
            except Exception:
                pass
            fname = f"{uuid.uuid4().hex}{ext}"
            upload_dir = os.path.join(str(settings.MEDIA_ROOT), 'uploads')
            os.makedirs(upload_dir, exist_ok=True)
            with open(os.path.join(upload_dir, fname), 'wb') as f:
                f.write(raw)
            saved_rel = os.path.join('uploads', fname)
        except Exception as e:
            logger.warning(f"Could not save upload: {e}")

        analysis = Analysis.objects.create(
            image=saved_rel,
            original_filename=getattr(image_file, 'name', ''),
            size_bytes=len(raw),
            content_type=getattr(image_file, 'content_type', ''),
            symptoms=symptoms,
            results={'predictions': predictions},
            quality_info=quality_info,
        )

        analysis_id = analysis.id

        # Filter out "Other / Unknown" predictions
        filtered_predictions = [
            pred for pred in predictions 
            if pred['disease'] not in ['Other / Unknown', 'Unknown', 'Other']
        ]
        
        # If all predictions were filtered out, keep at least one (but still filter Unknown)
        if not filtered_predictions and predictions:
            # Keep only non-Unknown predictions
            filtered_predictions = [
                pred for pred in predictions 
                if 'unknown' not in pred['disease'].lower() and 'other' not in pred['disease'].lower()
            ]
        
        # Get comprehensive disease info from service (only for filtered predictions)
        disease_info = {}
        for pred in filtered_predictions:
            disease_name = pred['disease']
            disease_info[disease_name] = {
                'desc': pred['description'],
                'advice': pred['treatment'],
                'urgency': pred['urgency'],
                'symptoms': pred['symptoms'],
                'prevention': pred['prevention']
            }
        
        return JsonResponse({
            'ok': True,
            'quality_ok': quality_ok,
            'quality_info': quality_info,
            'predictions': filtered_predictions,
            'symptoms': symptoms,
            'filename': getattr(image_file, 'name', None),
            'info': disease_info,
            'analysis_id': analysis_id,
            'user_message': user_message,
            'user_advice': user_advice,
            'category': category,
        })
        
    except Exception as e:
        # Log full traceback to server logs for diagnosis
        logger.exception('Prediction failed')
        return JsonResponse({
            'ok': False,
            'error': f'Prediction failed: {str(e)}',
            'quality_ok': False,
            'predictions': [],
            'symptoms': symptoms,
            'filename': getattr(image_file, 'name', None),
        }, status=500)


@login_required
def user_profile(request: HttpRequest) -> HttpResponse:
    # Handle profile updates (AJAX or multipart)
    if request.method == 'POST':
        content_type = str(request.content_type or '')
        try:
            profile, _ = Profile.objects.get_or_create(user=request.user)
            if 'multipart/form-data' in content_type:
                # avatar upload and basic fields
                if 'avatar' in request.FILES:
                    profile.avatar = request.FILES['avatar']
                profile.location = request.POST.get('location', profile.location)
                profile.bio = request.POST.get('bio', profile.bio)
                profile.save()
                return JsonResponse({'ok': True, 'message': 'Profile updated'})
            else:
                # JSON body
                body = json.loads(request.body.decode('utf-8')) if request.body else {}
                profile.location = body.get('location', profile.location)
                profile.bio = body.get('bio', profile.bio)
                profile.save()
                # Update user names/email optionally
                fn = body.get('first_name'); ln = body.get('last_name'); em = body.get('email')
                changed = False
                if isinstance(fn, str): request.user.first_name = fn; changed = True
                if isinstance(ln, str): request.user.last_name = ln; changed = True
                if isinstance(em, str): request.user.email = em; changed = True
                if changed:
                    request.user.save(update_fields=['first_name','last_name','email'])
                return JsonResponse({'ok': True, 'message': 'Profile updated'})
        except Exception as e:
            logger.exception('Profile update failed')
            return JsonResponse({'ok': False, 'error': str(e)}, status=500)

    # Render page
    profile, _ = Profile.objects.get_or_create(user=request.user)
    return render(request, 'user.html', {
        'user_obj': request.user,
        'profile': profile,
    })


@login_required
def change_password(request: HttpRequest) -> JsonResponse:
    if request.method != 'POST':
        return JsonResponse({'ok': False, 'error': 'Only POST allowed'}, status=405)
    try:
        body = json.loads(request.body.decode('utf-8')) if request.body else {}
        current = body.get('current_password')
        new = body.get('new_password')
        confirm = body.get('confirm_password')
        if not (current and new and confirm):
            return JsonResponse({'ok': False, 'error': 'All fields are required.'}, status=400)
        if not request.user.check_password(current):
            return JsonResponse({'ok': False, 'error': 'Current password is incorrect.'}, status=400)
        if new != confirm:
            return JsonResponse({'ok': False, 'error': 'Passwords do not match.'}, status=400)
        # Validate strength
        try:
            validate_password(new, user=request.user)
        except ValidationError as ve:
            return JsonResponse({'ok': False, 'error': ' '.join(ve.messages)}, status=400)
        request.user.set_password(new)
        request.user.save(update_fields=['password'])
        update_session_auth_hash(request, request.user)
        return JsonResponse({'ok': True, 'message': 'Password updated successfully'})
    except Exception as e:
        logger.exception('Password change failed')
        return JsonResponse({'ok': False, 'error': str(e)}, status=500)


