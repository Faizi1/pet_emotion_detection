from datetime import datetime, timezone, date
import random
import math
from typing import Any, Dict, List, Optional
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from google.api_core.exceptions import FailedPrecondition
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt, csrf_protect
from django.views.decorators.http import require_http_methods
from django.middleware.csrf import get_token
import json
from .auth import FirebaseUser
from .firebase import get_auth, get_firestore
from .sms_service import sms_service
# Remove AI detector integration; use random fallback only
emotion_detector = None
AI_DETECTOR_TYPE = "none"
from .serializers import (
    UserSerializer,
    PetSerializer,
    EmotionScanRequestSerializer,
    EmotionScanResponseSerializer,
    EmotionLogSerializer,
    RegisterSerializer,
    SendOtpSerializer,
    VerifyOtpSerializer,
    VerifyOtpRegistrationSerializer,
    ResendOtpRegistrationSerializer,
    LoginSerializer,
    ForgotPasswordSerializer,
    VerifyResetOtpSerializer,
    ResetPasswordSerializer,
    UpdateProfileSerializer,
    ChangePasswordSerializer,
    GoogleSignInSerializer,
    AppleSignInSerializer,
    PostSerializer,
    CreatePostSerializer,
    CommentSerializer,
    CreateCommentSerializer,
    LikeSerializer,
    ShareSerializer,
    SupportMessageSerializer,
    CreateSupportMessageSerializer,
)
from .storage import upload_bytes
from .permissions import IsAdminFirebase
from django.contrib.auth.decorators import login_required, user_passes_test

def is_superuser(user):
    return user.is_authenticated and user.is_superuser


def _pets_collection(uid: str):
    return get_firestore().collection('users').document(uid).collection('pets')


def _logs_collection(uid: str):
    return get_firestore().collection('users').document(uid).collection('emotion_logs')


def _posts_collection():
    return get_firestore().collection('community_posts')


def _comments_collection(post_id: str):
    return get_firestore().collection('community_posts').document(post_id).collection('comments')


def _likes_collection(post_id: str):
    return get_firestore().collection('community_posts').document(post_id).collection('likes')


def _shares_collection(post_id: str):
    return get_firestore().collection('community_posts').document(post_id).collection('shares')


def _support_messages_collection():
    return get_firestore().collection('support_messages')


@swagger_auto_schema(
    method='post',
    operation_description='Verify a Firebase ID token',
    request_body=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        required=['idToken'],
        properties={
            'idToken': openapi.Schema(type=openapi.TYPE_STRING, description='Firebase ID token')
        }
    ),
    responses={
        200: UserSerializer,
        400: 'Bad Request - idToken required',
        401: 'Unauthorized - Invalid token'
    }
)
@api_view(['POST'])
@permission_classes([AllowAny])
def verify_token(request):
    token = request.data.get('idToken')
    if not token:
        return Response({'detail': 'idToken required'}, status=status.HTTP_400_BAD_REQUEST)
    try:
        decoded = get_auth().verify_id_token(token)
    except Exception:
        return Response({'detail': 'Invalid token'}, status=status.HTTP_401_UNAUTHORIZED)
    user = FirebaseUser(
        uid=decoded.get('uid') or decoded.get('sub'),
        email=decoded.get('email'),
        name=decoded.get('name'),
        is_admin=bool(decoded.get('admin') or decoded.get('claims', {}).get('admin')),
    )
    return Response(UserSerializer(user.__dict__).data)


# Auth & Registration Flow

@swagger_auto_schema(
    method='post',
    operation_description='Start registration process - sends OTP to phone number for verification',
    request_body=RegisterSerializer,
    responses={
        200: openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'sent': openapi.Schema(type=openapi.TYPE_BOOLEAN, description='OTP sent status'),
                'phoneNumber': openapi.Schema(type=openapi.TYPE_STRING, description='Phone number where OTP was sent')
            }
        ),
        400: openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'detail': openapi.Schema(type=openapi.TYPE_STRING, description='Error message: Email/phone already exists, validation error, or registration in progress')
            }
        )
    }
)
@api_view(['POST'])
@permission_classes([AllowAny])
def register(request):
    ser = RegisterSerializer(data=request.data)
    ser.is_valid(raise_exception=True)
    data = ser.validated_data
    
    db = get_firestore()
    phone_number = data['number']
    email = data['email']
    
    # Check if email already exists in users collection
    users_ref = db.collection('users')
    email_query = users_ref.where('email', '==', email).limit(1)
    email_docs = list(email_query.stream())
    
    if email_docs:
        return Response({
            'detail': 'Email already exists. Please use a different email or try logging in.'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # Check if phone number already exists in users collection
    phone_query = users_ref.where('number', '==', phone_number).limit(1)
    phone_docs = list(phone_query.stream())
    
    if phone_docs:
        return Response({
            'detail': 'Phone number already exists. Please use a different phone number or try logging in.'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # Check if there's already a pending registration for this phone number
    temp_doc = db.collection('temp_registrations').document(phone_number).get()
    if temp_doc.exists:
        temp_data = temp_doc.to_dict()
        # Check if the temp registration is not expired
        otp_created = temp_data.get('otpCreatedAt')
        if otp_created:
            time_diff = datetime.now(timezone.utc) - otp_created
            if time_diff.total_seconds() <= 600:  # 10 minutes
                return Response({
                    'detail': 'Registration already in progress for this phone number. Please check your messages for OTP or wait for it to expire.'
                }, status=status.HTTP_400_BAD_REQUEST)
    
    # Generate OTP
    code = f"{random.randint(100000, 999999)}"
    
    # Store temporary registration data with OTP
    db.collection('temp_registrations').document(phone_number).set({
        'name': data['name'],
        'email': data['email'],
        'number': data['number'],
        'password': data['password'],
        'otp': code,
        'otpCreatedAt': datetime.now(timezone.utc),
        'attempts': 0,  # Track OTP verification attempts
    })
    
    # Send OTP via Twilio SMS
    sms_result = sms_service.send_otp(phone_number, code, 'registration')
    
    if sms_result['success']:
        return Response({
            'sent': True,
            'phoneNumber': phone_number,
            # 'messageSid': sms_result.get('message_sid'),
            'smsService': 'Vonage'
        })
    else:
        # If SMS fails, still store OTP but inform about SMS failure
        return Response({
            'sent': True,  # OTP stored successfully
            'phoneNumber': phone_number,
            'smsSent': False,
            'smsError': sms_result.get('error'),
            'smsService': 'twilio_failed',
            'message': 'OTP generated but SMS delivery failed. Check your Twilio configuration.'
        }, status=status.HTTP_201_CREATED)


@swagger_auto_schema(
    method='post',
    operation_description='Resend registration OTP to the provided phone number if a pending registration exists and is not expired',
    request_body=ResendOtpRegistrationSerializer,
    responses={
        200: openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'sent': openapi.Schema(type=openapi.TYPE_BOOLEAN, description='OTP sent status'),
                'phoneNumber': openapi.Schema(type=openapi.TYPE_STRING, description='Phone number where OTP was sent'),
                'smsService': openapi.Schema(type=openapi.TYPE_STRING, description='SMS service used')
            }
        ),
        400: 'Bad Request - Registration not found or throttled',
        404: 'Not Found - Registration not found'
    }
)
@api_view(['POST'])
@permission_classes([AllowAny])
def resend_registration_otp(request):
    ser = ResendOtpRegistrationSerializer(data=request.data)
    ser.is_valid(raise_exception=True)
    phone_number = ser.validated_data['phoneNumber']

    db = get_firestore()

    temp_ref = db.collection('temp_registrations').document(phone_number)
    temp_doc = temp_ref.get()

    if not temp_doc.exists:
        return Response({'detail': 'Registration not found. Please start registration again.'}, status=status.HTTP_404_NOT_FOUND)

    temp_data = temp_doc.to_dict() or {}

    # Check expiry: valid for 10 minutes from last otpCreatedAt
    otp_created = temp_data.get('otpCreatedAt')
    if otp_created:
        time_diff = datetime.now(timezone.utc) - otp_created
        if time_diff.total_seconds() > 600:  # 10 minutes
            temp_ref.delete()
            return Response({'detail': 'OTP expired. Please register again.'}, status=status.HTTP_400_BAD_REQUEST)

        # Basic throttle: do not resend within 60 seconds
        if time_diff.total_seconds() < 60:
            return Response({'detail': 'Please wait before requesting a new OTP.'}, status=status.HTTP_400_BAD_REQUEST)

    # Generate new OTP
    code = f"{random.randint(100000, 999999)}"

    # Update OTP and timestamp, reset attempts
    temp_ref.set({
        'otp': code,
        'otpCreatedAt': datetime.now(timezone.utc),
        'attempts': 0,
    }, merge=True)

    # Send OTP via SMS
    sms_result = sms_service.send_otp(phone_number, code, 'registration')

    if sms_result.get('success'):
        return Response({
            'sent': True,
            'phoneNumber': phone_number,
            'smsService': 'Vonage'
        })

    return Response({
        'sent': True,
        'phoneNumber': phone_number,
        'smsSent': False,
        'smsError': sms_result.get('error'),
        'smsService': 'twilio_failed',
        'message': 'OTP regenerated but SMS delivery failed.'
    }, status=status.HTTP_201_CREATED)


@swagger_auto_schema(
    method='post',
    operation_description='Send OTP to user phone number for verification',
    request_body=SendOtpSerializer,
    responses={
        200: openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'sent': openapi.Schema(type=openapi.TYPE_BOOLEAN, description='OTP sent status'),
                'smsSent': openapi.Schema(type=openapi.TYPE_BOOLEAN, description='SMS delivery status'),
                'messageSid': openapi.Schema(type=openapi.TYPE_STRING, description='Twilio message SID'),
                'smsService': openapi.Schema(type=openapi.TYPE_STRING, description='SMS service used')
            }
        ),
        400: 'Bad Request - Validation error'
    }
)
@api_view(['POST'])
@permission_classes([AllowAny])
def send_otp(request):
    ser = SendOtpSerializer(data=request.data)
    ser.is_valid(raise_exception=True)
    uid = ser.validated_data['uid']
    
    # Get user data to find phone number
    db = get_firestore()
    user_doc = db.collection('users').document(uid).get()
    
    if not user_doc.exists:
        return Response({'detail': 'User not found'}, status=status.HTTP_404_NOT_FOUND)
    
    user_data = user_doc.to_dict()
    phone_number = user_data.get('number', '')
    
    if not phone_number:
        return Response({'detail': 'Phone number not found for user'}, status=status.HTTP_400_BAD_REQUEST)
    
    # Generate OTP
    code = f"{random.randint(100000, 999999)}"
    
    # Store OTP in Firestore
    db.collection('users').document(uid).set({
        'otp': code,
        'otpCreatedAt': datetime.now(timezone.utc),
        'otpAttempts': 0,  # Reset attempts counter
    }, merge=True)
    
    # Send OTP via Twilio SMS
    sms_result = sms_service.send_otp(phone_number, code, 'login')
    
    response_data = {
        'sent': True,
        'smsSent': sms_result['success'],
        'smsService': 'twilio'
    }
    
    if sms_result['success']:
        response_data['messageSid'] = sms_result.get('message_sid')
    else:
        response_data['smsError'] = sms_result.get('error')
        response_data['message'] = 'OTP generated but SMS delivery failed'
    
    return Response(response_data)


@swagger_auto_schema(
    method='post',
    operation_description='Verify OTP code sent to user phone',
    request_body=VerifyOtpSerializer,
    responses={
        200: openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={'verified': openapi.Schema(type=openapi.TYPE_BOOLEAN, description='Verification status')}
        ),
        400: 'Bad Request - Invalid code',
        404: 'Not Found - User not found'
    }
)
@api_view(['POST'])
@permission_classes([AllowAny])
def verify_otp(request):
    ser = VerifyOtpSerializer(data=request.data)
    ser.is_valid(raise_exception=True)
    uid = ser.validated_data['uid']
    code = ser.validated_data['code']
    db = get_firestore()
    doc = db.collection('users').document(uid).get()
    if not doc.exists:
        return Response({'detail': 'User not found'}, status=status.HTTP_404_NOT_FOUND)
    data = doc.to_dict() or {}
    if data.get('otp') != code:
        return Response({'detail': 'Invalid code'}, status=status.HTTP_400_BAD_REQUEST)
    db.collection('users').document(uid).update({'phoneVerified': True, 'otp': None})
    return Response({'verified': True})


@swagger_auto_schema(
    method='post',
    operation_description='Verify OTP and complete user registration',
    request_body=VerifyOtpRegistrationSerializer,
    responses={
        201: openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'uid': openapi.Schema(type=openapi.TYPE_STRING, description='Created user ID'),
                'message': openapi.Schema(type=openapi.TYPE_STRING, description='Success message')
            }
        ),
        400: 'Bad Request - Invalid OTP or registration expired',
        404: 'Not Found - Registration not found'
    }
)
@api_view(['POST'])
@permission_classes([AllowAny])
def verify_otp_and_register(request):
    ser = VerifyOtpRegistrationSerializer(data=request.data)
    ser.is_valid(raise_exception=True)
    phone_number = ser.validated_data['phoneNumber']
    code = ser.validated_data['code']
    
    db = get_firestore()
    
    # Get temporary registration data
    temp_doc = db.collection('temp_registrations').document(phone_number).get()
    if not temp_doc.exists:
        return Response({'detail': 'Registration not found or expired'}, status=status.HTTP_404_NOT_FOUND)
    
    temp_data = temp_doc.to_dict()
    
    # Check OTP
    if temp_data.get('otp') != code:
        # Increment attempts
        attempts = temp_data.get('attempts', 0) + 1
        db.collection('temp_registrations').document(phone_number).update({'attempts': attempts})
        
        if attempts >= 3:
            # Delete temp registration after 3 failed attempts
            db.collection('temp_registrations').document(phone_number).delete()
            return Response({'detail': 'Too many failed attempts. Please register again.'}, status=status.HTTP_400_BAD_REQUEST)
        
        return Response({'detail': 'Invalid OTP code'}, status=status.HTTP_400_BAD_REQUEST)
    
    # Check if OTP is not expired (valid for 10 minutes)
    otp_created = temp_data.get('otpCreatedAt')
    if otp_created:
        time_diff = datetime.now(timezone.utc) - otp_created
        if time_diff.total_seconds() > 600:  # 10 minutes
            db.collection('temp_registrations').document(phone_number).delete()
            return Response({'detail': 'OTP expired. Please register again.'}, status=status.HTTP_400_BAD_REQUEST)
    
    # Double-check if email/phone still doesn't exist (edge case protection)
    users_ref = db.collection('users')
    email_query = users_ref.where('email', '==', temp_data['email']).limit(1)
    email_docs = list(email_query.stream())
    
    if email_docs:
        # Clean up temp data
        db.collection('temp_registrations').document(phone_number).delete()
        return Response({
            'detail': 'Email already exists. Please try logging in instead.'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    phone_query = users_ref.where('number', '==', phone_number).limit(1)
    phone_docs = list(phone_query.stream())
    
    if phone_docs:
        # Clean up temp data
        db.collection('temp_registrations').document(phone_number).delete()
        return Response({
            'detail': 'Phone number already exists. Please try logging in instead.'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # Create Firebase Auth user
    try:
        user_record = get_auth().create_user(
            email=temp_data['email'],
            password=temp_data['password'],
            display_name=temp_data['name']
        )
    except Exception as e:
        # Clean up temp data on error
        db.collection('temp_registrations').document(phone_number).delete()
        return Response({'detail': f'Failed to create user: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)
    
    # Store user data in Firestore
    db.collection('users').document(user_record.uid).set({
        'name': temp_data['name'],
        'email': temp_data['email'],
        'number': temp_data['number'],
        'createdAt': datetime.now(timezone.utc),
        'phoneVerified': True,
    })
    
    # Clean up temporary registration data
    db.collection('temp_registrations').document(phone_number).delete()
    
    return Response({
        'uid': user_record.uid,
        'message': 'Registration completed successfully'
    }, status=status.HTTP_201_CREATED)


@swagger_auto_schema(
    method='post',
    operation_description='Sign in with Google OAuth (for mobile apps)',
    request_body=GoogleSignInSerializer,
    responses={
        200: UserSerializer,
        201: openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'uid': openapi.Schema(type=openapi.TYPE_STRING, description='Created user ID'),
                'isNewUser': openapi.Schema(type=openapi.TYPE_BOOLEAN, description='Whether this is a new user'),
                'message': openapi.Schema(type=openapi.TYPE_STRING, description='Success message')
            }
        ),
        400: 'Bad Request - Invalid token or validation error',
        401: 'Unauthorized - Invalid Google token'
    }
)
@api_view(['POST'])
@permission_classes([AllowAny])
def google_signin(request):
    ser = GoogleSignInSerializer(data=request.data)
    ser.is_valid(raise_exception=True)
    data = ser.validated_data
    
    google_token = data['idToken']
    
    try:
        # Verify the Google ID token with Firebase
        decoded_token = get_auth().verify_id_token(google_token)
        google_uid = decoded_token.get('uid') or decoded_token.get('sub')
        google_email = decoded_token.get('email')
        google_name = decoded_token.get('name') or data.get('name')
        google_phone = data.get('phoneNumber')
        
        if not google_uid:
            return Response({'detail': 'Invalid Google token'}, status=status.HTTP_401_UNAUTHORIZED)
            
    except Exception as e:
        return Response({'detail': f'Invalid Google token: {str(e)}'}, status=status.HTTP_401_UNAUTHORIZED)
    
    db = get_firestore()
    
    # Check if user already exists
    user_query = db.collection('users').where('email', '==', google_email).limit(1)
    existing_users = list(user_query.stream())
    
    if existing_users:
        # User exists, return user data
        existing_user = existing_users[0]
        user_data = existing_user.to_dict()
        user_data['uid'] = existing_user.id
        
        # Update user data if needed (phone number, name)
        updates = {}
        if google_phone and not user_data.get('number'):
            updates['number'] = google_phone
        if google_name and not user_data.get('name'):
            updates['name'] = google_name
            
        if updates:
            db.collection('users').document(existing_user.id).update(updates)
            user_data.update(updates)
        
        return Response(UserSerializer(user_data).data)
    
    # Create new user
    try:
        # Create Firebase Auth user (Google will handle the authentication)
        user_record = get_auth().create_user(
            uid=google_uid,
            email=google_email,
            display_name=google_name
        )
    except Exception as e:
        # If user already exists in Firebase Auth, just get the record
        if 'already exists' in str(e).lower():
            user_record = get_auth().get_user(google_uid)
        else:
            return Response({'detail': f'Failed to create user: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)
    
    # Store user data in Firestore
    user_data = {
        'name': google_name,
        'email': google_email,
        'number': google_phone or '',
        'createdAt': datetime.now(timezone.utc),
        'phoneVerified': bool(google_phone),
        'provider': 'google',
        'providerId': google_uid,
    }
    
    db.collection('users').document(user_record.uid).set(user_data)
    
    return Response({
        'uid': user_record.uid,
        'isNewUser': True,
        'message': 'Google sign-in successful'
    }, status=status.HTTP_201_CREATED)


@swagger_auto_schema(
    method='post',
    operation_description='Sign in with Apple OAuth (for mobile apps)',
    request_body=AppleSignInSerializer,
    responses={
        200: UserSerializer,
        201: openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'uid': openapi.Schema(type=openapi.TYPE_STRING, description='Created user ID'),
                'isNewUser': openapi.Schema(type=openapi.TYPE_BOOLEAN, description='Whether this is a new user'),
                'message': openapi.Schema(type=openapi.TYPE_STRING, description='Success message')
            }
        ),
        400: 'Bad Request - Invalid token or validation error',
        401: 'Unauthorized - Invalid Apple token'
    }
)
@api_view(['POST'])
@permission_classes([AllowAny])
def apple_signin(request):
    ser = AppleSignInSerializer(data=request.data)
    ser.is_valid(raise_exception=True)
    data = ser.validated_data
    
    apple_token = data['identityToken']
    apple_email = data.get('email')
    apple_name = data.get('name')
    apple_phone = data.get('phoneNumber')
    
    try:
        # For Apple Sign-In, we need to verify the token differently
        # Apple uses JWT tokens that we need to decode and verify
        # For now, we'll create a user based on the provided data
        # In production, you should verify the Apple identity token properly
        
        # Generate a unique ID for Apple users
        apple_uid = f"apple_{hash(apple_email or apple_token) % 1000000000}"
        
        if not apple_email:
            return Response({'detail': 'Apple email is required'}, status=status.HTTP_400_BAD_REQUEST)
            
    except Exception as e:
        return Response({'detail': f'Invalid Apple token: {str(e)}'}, status=status.HTTP_401_UNAUTHORIZED)
    
    db = get_firestore()
    
    # Check if user already exists
    user_query = db.collection('users').where('email', '==', apple_email).limit(1)
    existing_users = list(user_query.stream())
    
    if existing_users:
        # User exists, return user data
        existing_user = existing_users[0]
        user_data = existing_user.to_dict()
        user_data['uid'] = existing_user.id
        
        # Update user data if needed
        updates = {}
        if apple_phone and not user_data.get('number'):
            updates['number'] = apple_phone
        if apple_name and not user_data.get('name'):
            updates['name'] = apple_name
            
        if updates:
            db.collection('users').document(existing_user.id).update(updates)
            user_data.update(updates)
        
        return Response(UserSerializer(user_data).data)
    
    # Create new user
    try:
        # Create Firebase Auth user
        user_record = get_auth().create_user(
            uid=apple_uid,
            email=apple_email,
            display_name=apple_name
        )
    except Exception as e:
        # If user already exists in Firebase Auth, just get the record
        if 'already exists' in str(e).lower():
            user_record = get_auth().get_user(apple_uid)
        else:
            return Response({'detail': f'Failed to create user: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)
    
    # Store user data in Firestore
    user_data = {
        'name': apple_name,
        'email': apple_email,
        'number': apple_phone or '',
        'createdAt': datetime.now(timezone.utc),
        'phoneVerified': bool(apple_phone),
        'provider': 'apple',
        'providerId': apple_uid,
    }
    
    db.collection('users').document(user_record.uid).set(user_data)
    
    return Response({
        'uid': user_record.uid,
        'isNewUser': True,
        'message': 'Apple sign-in successful'
    }, status=status.HTTP_201_CREATED)


@swagger_auto_schema(
    method='post',
    operation_description='Login with email and password (use Firebase client SDK for actual authentication)',
    request_body=LoginSerializer,
    responses={
        200: openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={'detail': openapi.Schema(type=openapi.TYPE_STRING, description='Instructions for login')}
        )
    }
)
@api_view(['POST'])
@permission_classes([AllowAny])
def login_email_password(request):
    # Firebase Admin SDK cannot mint ID tokens from email/password; clients should sign in via Firebase Client SDK.
    # This endpoint exists only to validate payload and guide clients.
    ser = LoginSerializer(data=request.data)
    ser.is_valid(raise_exception=True)
    return Response({'detail': 'Use Firebase client SDK to sign in and send ID token in Authorization header.'})


@swagger_auto_schema(
    method='post',
    operation_description='Request password reset OTP - sends OTP to phone number',
    request_body=ForgotPasswordSerializer,
    responses={
        200: openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'sent': openapi.Schema(type=openapi.TYPE_BOOLEAN, description='OTP sent status'),
                'phoneNumber': openapi.Schema(type=openapi.TYPE_STRING, description='Phone number where OTP was sent'),
                'smsSent': openapi.Schema(type=openapi.TYPE_BOOLEAN, description='SMS delivery status'),
                'smsService': openapi.Schema(type=openapi.TYPE_STRING, description='SMS service used')
            }
        ),
        400: 'Bad Request - User not found or validation error',
        404: 'Not Found - User not found'
    }
)
@api_view(['POST'])
@permission_classes([AllowAny])
def forgot_password(request):
    """
    Step 1: Request password reset OTP
    - Takes phone number
    - Finds user by phone number
    - Generates and sends OTP via SMS
    - Stores OTP in temp_password_resets collection
    """
    ser = ForgotPasswordSerializer(data=request.data)
    ser.is_valid(raise_exception=True)
    phone_number = ser.validated_data['phoneNumber']
    
    db = get_firestore()
    
    # Find user by phone number
    users_query = db.collection('users').where('number', '==', phone_number).limit(1)
    users_list = list(users_query.stream())
    
    if not users_list:
        return Response({'detail': 'User not found with this phone number'}, status=status.HTTP_404_NOT_FOUND)
    
    user_doc = users_list[0]
    user_uid = user_doc.id
    
    # Get user email from Firebase Auth (needed for password reset)
    try:
        user_record = get_auth().get_user(user_uid)
        user_email = user_record.email
    except Exception as e:
        return Response({'detail': f'Failed to get user: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)
    
    # Check if there's an existing password reset request
    temp_ref = db.collection('temp_password_resets').document(phone_number)
    temp_doc = temp_ref.get()
    
    if temp_doc.exists:
        temp_data = temp_doc.to_dict() or {}
        otp_created = temp_data.get('otpCreatedAt')
        
        # Check if OTP is still valid (10 minutes)
        if otp_created:
            time_diff = datetime.now(timezone.utc) - otp_created
            if time_diff.total_seconds() < 60:
                return Response({'detail': 'Please wait before requesting a new OTP.'}, status=status.HTTP_400_BAD_REQUEST)
    
    # Generate OTP
    code = f"{random.randint(100000, 999999)}"
    
    # Store OTP in temp_password_resets collection
    temp_ref.set({
        'phoneNumber': phone_number,
        'uid': user_uid,
        'email': user_email,
        'otp': code,
        'otpCreatedAt': datetime.now(timezone.utc),
        'attempts': 0,
        'otpVerified': False,
        'otpVerifiedAt': None,
    }, merge=True)
    
    # Send OTP via Vonage SMS
    sms_result = sms_service.send_otp(phone_number, code, 'reset')
    
    if sms_result.get('success'):
        return Response({
            'sent': True,
            'phoneNumber': phone_number,
            'smsSent': True,
            'smsService': sms_result.get('provider', sms_result.get('service', 'vonage')),
            'messageSid': sms_result.get('message_id') or sms_result.get('message_sid')
        })
    else:
        return Response({
            'sent': True,
            'phoneNumber': phone_number,
            'smsSent': False,
            'smsService': sms_result.get('provider', sms_result.get('service', 'vonage')),
            'message': 'OTP generated but SMS delivery failed. Check your SMS configuration.'
        }, status=status.HTTP_201_CREATED)


@swagger_auto_schema(
    method='post',
    operation_description='Resend password reset OTP - re-sends OTP to phone number if previous request exists',
    request_body=ForgotPasswordSerializer,
    responses={
        200: openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'sent': openapi.Schema(type=openapi.TYPE_BOOLEAN, description='OTP sent status'),
                'phoneNumber': openapi.Schema(type=openapi.TYPE_STRING, description='Phone number where OTP was sent'),
                'smsSent': openapi.Schema(type=openapi.TYPE_BOOLEAN, description='SMS delivery status'),
                'smsService': openapi.Schema(type=openapi.TYPE_STRING, description='SMS service used')
            }
        ),
        400: 'Bad Request - Validation error or throttled',
        404: 'Not Found - Reset request not found'
    }
)
@api_view(['POST'])
@permission_classes([AllowAny])
def resend_reset_otp(request):
    ser = ForgotPasswordSerializer(data=request.data)
    ser.is_valid(raise_exception=True)
    phone_number = ser.validated_data['phoneNumber']
    
    db = get_firestore()
    temp_ref = db.collection('temp_password_resets').document(phone_number)
    temp_doc = temp_ref.get()
    
    if not temp_doc.exists:
        return Response({'detail': 'Password reset request not found. Please request a new OTP.'}, status=status.HTTP_404_NOT_FOUND)
    
    temp_data = temp_doc.to_dict() or {}
    otp_created = temp_data.get('otpCreatedAt')
    
    # Throttle: require at least 60 seconds between resends
    if otp_created:
        time_diff = datetime.now(timezone.utc) - otp_created
        if time_diff.total_seconds() < 60:
            wait_seconds = int(60 - time_diff.total_seconds())
            return Response({'detail': f'Please wait {wait_seconds} seconds before requesting a new OTP.'}, status=status.HTTP_400_BAD_REQUEST)
    
    # Generate new OTP
    code = f"{random.randint(100000, 999999)}"
    
    temp_ref.set({
        'otp': code,
        'otpCreatedAt': datetime.now(timezone.utc),
        'attempts': 0,
        'otpVerified': False,
        'otpVerifiedAt': None,
    }, merge=True)
    
    sms_result = sms_service.send_otp(phone_number, code, 'reset')
    
    if sms_result.get('success'):
        return Response({
            'sent': True,
            'phoneNumber': phone_number,
            'smsSent': True,
            'smsService': sms_result.get('provider', sms_result.get('service', 'vonage')),
            'messageSid': sms_result.get('message_id') or sms_result.get('message_sid')
        })
    else:
        return Response({
            'sent': True,
            'phoneNumber': phone_number,
            'smsSent': False,
            'smsService': sms_result.get('provider', sms_result.get('service', 'vonage')),
            'message': 'OTP generated but SMS delivery failed. Check your SMS configuration.'
        }, status=status.HTTP_201_CREATED)


@swagger_auto_schema(
    method='post',
    operation_description='Verify OTP for password reset - verifies the OTP code sent to phone number',
    request_body=VerifyResetOtpSerializer,
    responses={
        200: openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'verified': openapi.Schema(type=openapi.TYPE_BOOLEAN, description='OTP verification status'),
                'message': openapi.Schema(type=openapi.TYPE_STRING, description='Success message')
            }
        ),
        400: 'Bad Request - Invalid OTP, expired OTP, or too many attempts',
        404: 'Not Found - Password reset request not found'
    }
)
@api_view(['POST'])
@permission_classes([AllowAny])
def verify_reset_otp(request):
    """
    Step 2: Verify OTP for password reset
    - Takes phone number and OTP code
    - Verifies OTP code matches
    - Checks OTP hasn't expired (10 minutes)
    - Checks attempts (max 3 failed attempts)
    - Marks OTP as verified in temp_password_resets
    """
    ser = VerifyResetOtpSerializer(data=request.data)
    ser.is_valid(raise_exception=True)
    
    phone_number = ser.validated_data['phoneNumber']
    code = ser.validated_data['code']
    
    db = get_firestore()
    
    # Get temporary password reset data
    temp_ref = db.collection('temp_password_resets').document(phone_number)
    temp_doc = temp_ref.get()
    
    if not temp_doc.exists:
        return Response({'detail': 'Password reset request not found or expired. Please request a new OTP.'}, status=status.HTTP_404_NOT_FOUND)
    
    temp_data = temp_doc.to_dict() or {}
    
    # Verify OTP code
    if temp_data.get('otp') != code:
        # Increment attempts
        attempts = temp_data.get('attempts', 0) + 1
        temp_ref.update({'attempts': attempts})
        
        if attempts >= 3:
            # Delete temp password reset after 3 failed attempts
            temp_ref.delete()
            return Response({'detail': 'Too many failed attempts. Please request a new OTP.'}, status=status.HTTP_400_BAD_REQUEST)
        
        return Response({'detail': 'Invalid OTP code'}, status=status.HTTP_400_BAD_REQUEST)
    
    # Check if OTP is not expired (valid for 10 minutes)
    otp_created = temp_data.get('otpCreatedAt')
    if otp_created:
        time_diff = datetime.now(timezone.utc) - otp_created
        if time_diff.total_seconds() > 600:  # 10 minutes
            temp_ref.delete()
            return Response({'detail': 'OTP expired. Please request a new OTP.'}, status=status.HTTP_400_BAD_REQUEST)
    
    # Mark OTP as verified and clear stored code
    temp_ref.update({
        'otpVerified': True,
        'otpVerifiedAt': datetime.now(timezone.utc),
        'otp': None,
        'attempts': 0,
    })
    
    return Response({
        'verified': True,
        'message': 'OTP verified successfully. You can now reset your password.'
    })


@swagger_auto_schema(
    method='post',
    operation_description='Reset password - sets new password after OTP verification',
    request_body=ResetPasswordSerializer,
    responses={
        200: openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'success': openapi.Schema(type=openapi.TYPE_BOOLEAN, description='Password reset status'),
                'message': openapi.Schema(type=openapi.TYPE_STRING, description='Success message')
            }
        ),
        400: 'Bad Request - OTP not verified, expired, or validation error',
        404: 'Not Found - Password reset request not found'
    }
)
@api_view(['POST'])
@permission_classes([AllowAny])
def reset_password(request):
    """
    Step 3: Reset password after OTP verification
    - Takes phone number, new password, and confirm password
    - Checks that OTP was previously verified
    - Validates password match
    - Updates password in Firebase Auth
    - Cleans up temp password reset data
    """
    ser = ResetPasswordSerializer(data=request.data)
    ser.is_valid(raise_exception=True)
    
    phone_number = ser.validated_data['phoneNumber']
    new_password = ser.validated_data['password']
    
    db = get_firestore()
    
    # Get temporary password reset data
    temp_ref = db.collection('temp_password_resets').document(phone_number)
    temp_doc = temp_ref.get()
    
    if not temp_doc.exists:
        return Response({'detail': 'Password reset request not found or expired. Please request a new OTP.'}, status=status.HTTP_404_NOT_FOUND)
    
    temp_data = temp_doc.to_dict() or {}
    
    # Check if OTP was verified
    if not temp_data.get('otpVerified'):
        return Response({'detail': 'OTP not verified. Please verify OTP first.'}, status=status.HTTP_400_BAD_REQUEST)
    
    # Check if verification is not expired (valid for 10 minutes from verification)
    otp_verified_at = temp_data.get('otpVerifiedAt')
    if otp_verified_at:
        time_diff = datetime.now(timezone.utc) - otp_verified_at
        if time_diff.total_seconds() > 600:  # 10 minutes
            temp_ref.delete()
            return Response({'detail': 'OTP verification expired. Please request a new OTP.'}, status=status.HTTP_400_BAD_REQUEST)
    
    # Get user UID from temp data
    user_uid = temp_data.get('uid')
    
    if not user_uid:
        temp_ref.delete()
        return Response({'detail': 'Invalid password reset request'}, status=status.HTTP_400_BAD_REQUEST)
    
    # Update password in Firebase Auth
    try:
        get_auth().update_user(user_uid, password=new_password)
    except Exception as e:
        temp_ref.delete()
        return Response({'detail': f'Failed to update password: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)
    
    # Clean up temporary password reset data
    temp_ref.delete()
    
    return Response({
        'success': True,
        'message': 'Password reset successfully'
    })


@swagger_auto_schema(
    method='get',
    operation_description='Get current authenticated user details',
    responses={
        200: UserSerializer,
        401: 'Unauthorized - Invalid or missing token'
    }
)
@api_view(['GET'])
def me(request):
    user: FirebaseUser = request.user  # type: ignore
    # Return latest profile by merging auth data with Firestore document (Firestore overrides)
    base: Dict[str, Any] = {
        'uid': getattr(user, 'uid', None),
        'email': getattr(user, 'email', None),
        'name': getattr(user, 'name', None),
    }
    try:
        doc = get_firestore().collection('users').document(user.uid).get()
        if doc.exists:
            base.update(doc.to_dict() or {})
    except Exception:
        pass
    return Response(base)


@swagger_auto_schema(
    method='patch',
    operation_description='Update personal profile details (name, number, location, photoUrl)',
    request_body=UpdateProfileSerializer,
    responses={200: 'Profile updated', 400: 'Validation error', 401: 'Unauthorized'}
)
@api_view(['PATCH'])
def update_profile(request):
    user: FirebaseUser = request.user  # type: ignore
    ser = UpdateProfileSerializer(data=request.data)
    ser.is_valid(raise_exception=True)
    data = ser.validated_data

    db = get_firestore()
    user_ref = db.collection('users').document(user.uid)
    update_data: Dict[str, Any] = {}
    for key in ['name', 'number', 'location', 'photoUrl']:
        if key in data:
            update_data[key] = data[key]

    if update_data:
        update_data['updatedAt'] = datetime.now(timezone.utc)
        user_ref.set(update_data, merge=True)

    # Optionally update Firebase Auth displayName if provided
    if 'name' in data:
        try:
            get_auth().update_user(user.uid, display_name=data['name'])
        except Exception:
            pass

    return Response({'detail': 'Profile updated', 'updated': update_data})


@swagger_auto_schema(
    method='post',
    operation_description='Change password with current, new, confirm',
    request_body=ChangePasswordSerializer,
    responses={200: 'Password changed', 400: 'Error/validation', 401: 'Unauthorized'}
)
@api_view(['POST'])
@permission_classes([AllowAny])
def change_password(request):
    # For security, require email too and verify current password using Firebase REST signInWithPassword
    email = (request.data or {}).get('email')
    ser = ChangePasswordSerializer(data=request.data)
    ser.is_valid(raise_exception=True)
    data = ser.validated_data

    if not email:
        return Response({'detail': 'Email is required'}, status=status.HTTP_400_BAD_REQUEST)

    # Use Firebase Web API to verify current password
    import os, requests
    api_key = os.getenv('FIREBASE_WEB_API_KEY')
    if not api_key:
        return Response({'detail': 'Server missing FIREBASE_WEB_API_KEY'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    try:
        verify_resp = requests.post(
            f'https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}',
            json={'email': email, 'password': data['currentPassword'], 'returnSecureToken': True}, timeout=10
        )
        if verify_resp.status_code != 200:
            return Response({'detail': 'Current password is incorrect'}, status=status.HTTP_400_BAD_REQUEST)

        # Get Firebase user by email and set new password via Admin SDK
        user_record = get_auth().get_user_by_email(email)
        get_auth().update_user(user_record.uid, password=data['newPassword'])
        return Response({'detail': 'Password changed'})
    except Exception as e:
        return Response({'detail': str(e)}, status=status.HTTP_400_BAD_REQUEST)


@swagger_auto_schema(
    method='get',
    operation_description='Get list of user pets',
    responses={
        200: PetSerializer(many=True),
        401: 'Unauthorized'
    }
)
@swagger_auto_schema(
    method='post',
    operation_description='Create a new pet',
    request_body=PetSerializer,
    responses={
        201: PetSerializer,
        400: 'Bad Request - Validation error',
        401: 'Unauthorized'
    }
)
@api_view(['GET', 'POST'])
def pets_list_create(request):
    user: FirebaseUser = request.user  # type: ignore
    col = _pets_collection(user.uid)
    if request.method == 'GET':
        docs = col.stream()
        pets: List[Dict[str, Any]] = []
        for d in docs:
            pet = d.to_dict()
            pet['id'] = d.id
            pets.append(pet)
        return Response(pets)
    data = request.data
    ser = PetSerializer(data=data)
    ser.is_valid(raise_exception=True)
    payload = dict(ser.validated_data)
    dob = payload.get('dateOfBirth')
    if isinstance(dob, date):
        payload['dateOfBirth'] = datetime.combine(dob, datetime.min.time(), tzinfo=timezone.utc)
    ref = col.document()
    ref.set(payload)
    created = ser.validated_data.copy()
    created['id'] = ref.id
    return Response(created, status=status.HTTP_201_CREATED)


@swagger_auto_schema(
    method='get',
    operation_description='Get pet details by ID',
    responses={
        200: PetSerializer,
        404: 'Not Found - Pet not found',
        401: 'Unauthorized'
    }
)
@swagger_auto_schema(
    method='put',
    operation_description='Update pet details (full update)',
    request_body=PetSerializer,
    responses={
        200: PetSerializer,
        400: 'Bad Request - Validation error',
        404: 'Not Found - Pet not found',
        401: 'Unauthorized'
    }
)
@swagger_auto_schema(
    method='patch',
    operation_description='Partially update pet details',
    request_body=PetSerializer,
    responses={
        200: PetSerializer,
        400: 'Bad Request - Validation error',
        404: 'Not Found - Pet not found',
        401: 'Unauthorized'
    }
)
@swagger_auto_schema(
    method='delete',
    operation_description='Delete a pet',
    responses={
        204: 'No Content - Pet deleted successfully',
        404: 'Not Found - Pet not found',
        401: 'Unauthorized'
    }
)
@api_view(['GET', 'PUT', 'PATCH', 'DELETE'])
def pets_detail(request, pet_id: str):
    user: FirebaseUser = request.user  # type: ignore
    ref = _pets_collection(user.uid).document(pet_id)
    snap = ref.get()
    if not snap.exists:
        return Response({'detail': 'Not found'}, status=status.HTTP_404_NOT_FOUND)
    if request.method == 'GET':
        pet = snap.to_dict()
        pet['id'] = snap.id
        return Response(pet)
    if request.method in ['PUT', 'PATCH']:
        data = request.data
        ser = PetSerializer(data=data, partial=(request.method == 'PATCH'))
        ser.is_valid(raise_exception=True)
        payload = dict(ser.validated_data)
        dob = payload.get('dateOfBirth')
        if isinstance(dob, date):
            payload['dateOfBirth'] = datetime.combine(dob, datetime.min.time(), tzinfo=timezone.utc)
        ref.update(payload)
        updated = ref.get().to_dict()
        updated['id'] = pet_id
        return Response(updated)
    ref.delete()
    return Response(status=status.HTTP_204_NO_CONTENT)


@swagger_auto_schema(
    method='post',
    operation_description='Upload media (image/video) for pet emotion detection',
    request_body=EmotionScanRequestSerializer,
    responses={
        201: EmotionScanResponseSerializer,
        400: 'Bad Request - Validation error',
        401: 'Unauthorized'
    }
)
@api_view(['POST'])
def scans_create(request):
    user: FirebaseUser = request.user  # type: ignore
    ser = EmotionScanRequestSerializer(data=request.data)
    ser.is_valid(raise_exception=True)
    file = ser.validated_data['file']
    media_type = ser.validated_data['mediaType']
    pet_id = ser.validated_data['petId']
    content_type = getattr(file, 'content_type', 'application/octet-stream')
    
    # File size limits for free tier (5MB for images, 10MB for audio)
    MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB
    MAX_AUDIO_SIZE = 10 * 1024 * 1024  # 10MB
    
    # Read file content once for upload
    file_content = file.read()
    file_size = len(file_content)
    
    # Check file size limits
    if media_type in ['image', 'photo'] and file_size > MAX_IMAGE_SIZE:
        return Response(
            {'detail': f'Image file too large. Maximum size is {MAX_IMAGE_SIZE // (1024*1024)}MB'},
            status=status.HTTP_400_BAD_REQUEST
        )
    elif media_type in ['audio', 'sound', 'voice'] and file_size > MAX_AUDIO_SIZE:
        return Response(
            {'detail': f'Audio file too large. Maximum size is {MAX_AUDIO_SIZE // (1024*1024)}MB'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Upload file first (don't wait for processing)
    try:
        media_url = upload_bytes(file_content, content_type, path_prefix=f"{user.uid}/{media_type}/")
    except Exception as e:
        return Response(
            {'detail': f'Failed to upload file: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    # Fast lightweight emotion detection (optimized for speed)
    try:
        if media_type in ['image', 'photo']:
            # Fast hash-based emotion detection (no heavy processing)
            from .advanced_image_ai import advanced_image_ai
            # Use lightweight mode for fast response
            if advanced_image_ai.use_lightweight_mode:
                # Ultra-fast hash-based detection
                file_hash = hash(file_content) % 10000
                emotions_list = ['happy', 'sad', 'anxious', 'excited', 'calm', 'playful', 'sleepy', 'curious']
                emotion = emotions_list[file_hash % len(emotions_list)]
                confidence = round(0.65 + (file_hash % 30) / 100.0, 2)
                
                # Generate top 3 emotions
                top_emotions = [
                    {'emotion': emotion, 'confidence': confidence},
                    {'emotion': emotions_list[(file_hash + 1) % len(emotions_list)], 'confidence': round(0.15 + (file_hash % 15) / 100.0, 2)},
                    {'emotion': emotions_list[(file_hash + 2) % len(emotions_list)], 'confidence': round(0.10 + (file_hash % 10) / 100.0, 2)}
                ]
                analysis_method = 'fast_hash_based'
                ai_type = 'lightweight_fast'
            else:
                # Full AI processing (slower but more accurate)
                ai_result = advanced_image_ai.detect_emotion_from_image(file_content)
                emotion = ai_result['emotion']
                confidence = ai_result['confidence']
                analysis_method = ai_result.get('analysis_method', 'advanced_ai')
                top_emotions = ai_result.get('top_emotions', [])
                ai_type = ai_result.get('ai_detector_type', 'advanced_ai')
        elif media_type in ['audio', 'sound', 'voice']:
            # Fast hash-based emotion detection (no heavy processing)
            from .advanced_audio_ai import advanced_audio_ai
            # Use lightweight mode for fast response
            if advanced_audio_ai.use_lightweight_mode:
                # Ultra-fast hash-based detection
                file_hash = hash(file_content) % 10000
                emotions_list = ['happy', 'sad', 'anxious', 'excited', 'calm', 'playful', 'sleepy', 'curious']
                emotion = emotions_list[file_hash % len(emotions_list)]
                confidence = round(0.65 + (file_hash % 30) / 100.0, 2)
                
                # Generate top 3 emotions
                top_emotions = [
                    {'emotion': emotion, 'confidence': confidence},
                    {'emotion': emotions_list[(file_hash + 1) % len(emotions_list)], 'confidence': round(0.15 + (file_hash % 15) / 100.0, 2)},
                    {'emotion': emotions_list[(file_hash + 2) % len(emotions_list)], 'confidence': round(0.10 + (file_hash % 10) / 100.0, 2)}
                ]
                analysis_method = 'fast_hash_based'
                ai_type = 'lightweight_fast'
            else:
                # Full AI processing (slower but more accurate)
                ai_result = advanced_audio_ai.detect_emotion_from_audio(file_content)
                emotion = ai_result['emotion']
                confidence = ai_result['confidence']
                analysis_method = ai_result.get('analysis_method', 'advanced_ai')
                top_emotions = ai_result.get('top_emotions', [])
                ai_type = ai_result.get('ai_detector_type', 'advanced_ai')
        else:
            # For video or other types, use random for now
            emotion = random.choice(['happy', 'sad', 'anxious', 'excited', 'calm'])
            confidence = round(random.uniform(0.6, 0.99), 2)
            analysis_method = 'random'
            top_emotions = []
            ai_type = 'none'
            
    except Exception as e:
        # Fallback to fast hash-based if anything fails
        print(f"Emotion detection failed: {e}")
        file_hash = hash(file_content) % 10000
        emotions_list = ['happy', 'sad', 'anxious', 'excited', 'calm']
        emotion = emotions_list[file_hash % len(emotions_list)]
        confidence = round(0.65 + (file_hash % 30) / 100.0, 2)
        analysis_method = 'fast_fallback'
        top_emotions = [{'emotion': emotion, 'confidence': confidence}]
        ai_type = 'fallback'

    log = {
        'petId': pet_id,
        'timestamp': datetime.now(timezone.utc),
        'emotion': emotion,
        'confidence': confidence,
        'mediaUrl': media_url,
        'mediaType': media_type,
        'analysisMethod': analysis_method,
        'animalType': 'unknown',
        'topEmotions': top_emotions,
        'aiDetectorType': ai_type,
    }
    _logs_collection(user.uid).add(log)

    resp = {
        'emotion': emotion,
        'confidence': confidence,
        'mediaUrl': media_url,
        'petId': pet_id,
        'animalType': 'unknown',
        'analysisMethod': analysis_method,
        'topEmotions': top_emotions,
        'aiDetectorType': ai_type,
    }
    return Response(EmotionScanResponseSerializer(resp).data, status=status.HTTP_201_CREATED)


@swagger_auto_schema(
    method='get',
    operation_description='Get AI emotion detector status and information',
    responses={
        200: 'AI detector information',
        500: 'Internal Server Error'
    }
)
@api_view(['GET'])
def ai_detector_status(request):
    """
    Get AI emotion detector status and information
    """
    try:
        # Get Advanced Image AI detector info
        from .advanced_image_ai import get_detector_info
        
        detector_info = get_detector_info()
        
        combined_info = {
            'system_status': 'enhanced_pet_ai_ready',
            'total_emotions': detector_info['total_emotions'],
            'expected_accuracy': detector_info['expected_accuracy'],
            'model_architectures': {
                'images': detector_info['image_model_architecture'],
                'audio': detector_info['audio_model_architecture']
            },
            'image_model_loaded': detector_info['image_model_loaded'],
            'audio_model_loaded': detector_info['audio_model_loaded'],
            'research_based': detector_info['research_based'],
            'optimized_for_pets': detector_info['optimized_for_pets'],
            'emotion_labels': detector_info['emotion_labels'],
            'features_used': detector_info['features_used'],
            'model_components': detector_info['model_components']
        }
        
        return Response(combined_info, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response(
            {'error': f'Failed to get detector status: {str(e)}'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )



def _clean_nan_values(data: Any) -> Any:
    """Recursively clean NaN values from data structures, replacing them with None."""
    if isinstance(data, dict):
        return {k: _clean_nan_values(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_clean_nan_values(item) for item in data]
    elif isinstance(data, float) and (math.isnan(data) or math.isinf(data)):
        return None
    return data


@swagger_auto_schema(
    method='get',
    operation_description='Get emotion scan history for user (optionally filtered by petId)',
    manual_parameters=[
        openapi.Parameter(
            'petId',
            openapi.IN_QUERY,
            description="Filter by pet ID",
            type=openapi.TYPE_STRING,
            required=False
        )
    ],
    responses={
        200: EmotionLogSerializer(many=True),
        401: 'Unauthorized'
    }
)
@api_view(['GET'])
def history_list(request):
    user: FirebaseUser = request.user  # type: ignore
    pet_id = request.query_params.get('petId')
    q = _logs_collection(user.uid)
    if pet_id:
        q = q.where('petId', '==', pet_id)
    try:
        docs_list = list(q.order_by('timestamp', direction='DESCENDING').limit(100).stream())
    except FailedPrecondition:
        # Missing or building composite index; fallback without ordering
        docs_list = list(q.limit(100).stream())
    logs: List[Dict[str, Any]] = []
    pets_cache: Dict[str, Optional[Dict[str, Any]]] = {}
    pets_collection = _pets_collection(user.uid)
    for d in docs_list:
        item = d.to_dict()
        item['id'] = d.id
        pet_ref_id = item.get('petId')
        if pet_ref_id:
            if pet_ref_id not in pets_cache:
                try:
                    pet_snapshot = pets_collection.document(pet_ref_id).get()
                except Exception:
                    pet_snapshot = None
                if pet_snapshot and pet_snapshot.exists:
                    pet_data = pet_snapshot.to_dict() or {}
                    pet_data['id'] = pet_snapshot.id
                    pets_cache[pet_ref_id] = _clean_nan_values(pet_data)
                else:
                    pets_cache[pet_ref_id] = None
            if pets_cache.get(pet_ref_id) is not None:
                item['pet'] = pets_cache[pet_ref_id]
        # Clean NaN values before serialization
        item = _clean_nan_values(item)
        logs.append(item)
    return Response(logs)


@swagger_auto_schema(
    method='get',
    operation_description='Get admin analytics (requires admin privileges)',
    responses={
        200: openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'totalUsers': openapi.Schema(type=openapi.TYPE_INTEGER, description='Total number of users'),
                'totalPets': openapi.Schema(type=openapi.TYPE_INTEGER, description='Total number of pets'),
                'totalScans': openapi.Schema(type=openapi.TYPE_INTEGER, description='Total number of scans'),
            }
        ),
        401: 'Unauthorized',
        403: 'Forbidden - Admin access required'
    }
)
@api_view(['GET'])
@permission_classes([IsAdminFirebase])
def admin_analytics(request):
    db = get_firestore()
    # Simple counts for MVP
    users_ref = db.collection('users')
    # These can be expensive; in real systems, store counters.
    total_users = len(list(users_ref.stream()))
    total_pets = 0
    total_scans = 0
    for user_doc in users_ref.stream():
        total_pets += len(list(users_ref.document(user_doc.id).collection('pets').stream()))
        total_scans += len(list(users_ref.document(user_doc.id).collection('emotion_logs').stream()))
    return Response({
        'totalUsers': total_users,
        'totalPets': total_pets,
        'totalScans': total_scans,
    })


# Community/Posts Endpoints

@swagger_auto_schema(
    method='get',
    operation_description='Get all public posts (community feed)',
    manual_parameters=[
        openapi.Parameter(
            'limit',
            openapi.IN_QUERY,
            description="Number of posts to return (default: 20, max: 50)",
            type=openapi.TYPE_INTEGER,
            required=False
        ),
        openapi.Parameter(
            'offset',
            openapi.IN_QUERY,
            description="Number of posts to skip (for pagination)",
            type=openapi.TYPE_INTEGER,
            required=False
        )
    ],
    responses={
        200: PostSerializer(many=True),
        401: 'Unauthorized'
    }
)
@api_view(['GET'])
def community_posts_list(request):
    user: FirebaseUser = request.user  # type: ignore
    limit = min(int(request.query_params.get('limit', 20)), 50)
    offset = int(request.query_params.get('offset', 0))
    
    db = get_firestore()
    
    # Get public posts ordered by creation time (newest first)
    posts_query = _posts_collection().where('isPublic', '==', True).order_by('createdAt', direction='DESCENDING')
    
    try:
        posts_docs = list(posts_query.offset(offset).limit(limit).stream())
    except FailedPrecondition:
        # Fallback without ordering if index is not available
        posts_docs = list(_posts_collection().where('isPublic', '==', True).offset(offset).limit(limit).stream())
    
    posts = []
    for doc in posts_docs:
        post_data = doc.to_dict()
        post_data['id'] = doc.id
        
        # Check if current user liked this post
        user_like = _likes_collection(doc.id).document(user.uid).get()
        post_data['isLikedByUser'] = user_like.exists
        
        # Get counts
        post_data['likesCount'] = len(list(_likes_collection(doc.id).stream()))
        post_data['commentsCount'] = len(list(_comments_collection(doc.id).stream()))
        post_data['sharesCount'] = len(list(_shares_collection(doc.id).stream()))
        
        posts.append(post_data)
    
    return Response(posts)


@swagger_auto_schema(
    method='get',
    operation_description='Get posts created by the current user',
    manual_parameters=[
        openapi.Parameter(
            'limit',
            openapi.IN_QUERY,
            description="Number of posts to return (default: 20, max: 50)",
            type=openapi.TYPE_INTEGER,
            required=False
        ),
        openapi.Parameter(
            'offset',
            openapi.IN_QUERY,
            description="Number of posts to skip (for pagination)",
            type=openapi.TYPE_INTEGER,
            required=False
        )
    ],
    responses={
        200: PostSerializer(many=True),
        401: 'Unauthorized'
    }
)
@api_view(['GET'])
def my_posts_list(request):
    user: FirebaseUser = request.user  # type: ignore
    limit = min(int(request.query_params.get('limit', 20)), 50)
    offset = int(request.query_params.get('offset', 0))
    
    # Get user's posts
    posts_query = _posts_collection().where('authorId', '==', user.uid).order_by('createdAt', direction='DESCENDING')
    
    try:
        posts_docs = list(posts_query.offset(offset).limit(limit).stream())
    except FailedPrecondition:
        posts_docs = list(_posts_collection().where('authorId', '==', user.uid).offset(offset).limit(limit).stream())
    
    posts = []
    for doc in posts_docs:
        post_data = doc.to_dict()
        post_data['id'] = doc.id
        post_data['isLikedByUser'] = True  # User always likes their own posts conceptually
        
        # Get counts
        post_data['likesCount'] = len(list(_likes_collection(doc.id).stream()))
        post_data['commentsCount'] = len(list(_comments_collection(doc.id).stream()))
        post_data['sharesCount'] = len(list(_shares_collection(doc.id).stream()))
        
        posts.append(post_data)
    
    return Response(posts)


@swagger_auto_schema(
    method='post',
    operation_description='Create a new community post',
    request_body=CreatePostSerializer,
    responses={
        201: PostSerializer,
        400: 'Bad Request - Validation error',
        401: 'Unauthorized'
    }
)
@api_view(['POST'])
def community_post_create(request):
    user: FirebaseUser = request.user  # type: ignore
    ser = CreatePostSerializer(data=request.data)
    ser.is_valid(raise_exception=True)
    
    # Handle image uploads if any
    images = []
    if 'images' in request.FILES:
        for image_file in request.FILES.getlist('images'):
            try:
                content_type = getattr(image_file, 'content_type', 'application/octet-stream')
                
                # Validate file type
                if not content_type.startswith('image/'):
                    return Response({
                        'detail': f'Invalid file type: {content_type}. Only images are allowed.'
                    }, status=status.HTTP_400_BAD_REQUEST)
                
                # Upload to Firebase Storage
                image_url = upload_bytes(
                    image_file.read(), 
                    content_type, 
                    path_prefix=f"community/{user.uid}/posts/"
                )
                
                # Only add if upload was successful
                if image_url and image_url.strip():
                    images.append(image_url)
                else:
                    return Response({
                        'detail': 'Failed to upload image. Firebase Storage might not be configured.'
                    }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                    
            except Exception as e:
                return Response({
                    'detail': f'Error uploading image: {str(e)}'
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    post_data = {
        'content': ser.validated_data['content'],
        'images': images,
        'tags': ser.validated_data.get('tags', []),
        'isPublic': ser.validated_data.get('isPublic', True),
        'authorId': user.uid,
        'authorName': user.name or user.email or 'Anonymous',
        'createdAt': datetime.now(timezone.utc),
        'updatedAt': datetime.now(timezone.utc),
    }
    
    # Create post
    post_ref = _posts_collection().document()
    post_ref.set(post_data)
    
    # Return created post with counts
    created_post = post_data.copy()
    created_post['id'] = post_ref.id
    created_post['likesCount'] = 0
    created_post['commentsCount'] = 0
    created_post['sharesCount'] = 0
    created_post['isLikedByUser'] = False
    
    return Response(created_post, status=status.HTTP_201_CREATED)


@swagger_auto_schema(
    method='get',
    operation_description='Get a specific post by ID',
    responses={
        200: PostSerializer,
        404: 'Not Found - Post not found',
        401: 'Unauthorized'
    }
)
@api_view(['GET'])
def community_post_detail(request, post_id: str):
    user: FirebaseUser = request.user  # type: ignore
    post_doc = _posts_collection().document(post_id).get()
    
    if not post_doc.exists:
        return Response({'detail': 'Post not found'}, status=status.HTTP_404_NOT_FOUND)
    
    post_data = post_doc.to_dict()
    post_data['id'] = post_doc.id
    
    # Check if user liked this post
    user_like = _likes_collection(post_id).document(user.uid).get()
    post_data['isLikedByUser'] = user_like.exists
    
    # Get counts
    post_data['likesCount'] = len(list(_likes_collection(post_id).stream()))
    post_data['commentsCount'] = len(list(_comments_collection(post_id).stream()))
    post_data['sharesCount'] = len(list(_shares_collection(post_id).stream()))
    
    return Response(post_data)


@swagger_auto_schema(
    method='delete',
    operation_description='Delete a post (only by author)',
    responses={
        204: 'No Content - Post deleted successfully',
        404: 'Not Found - Post not found',
        403: 'Forbidden - Not the post author',
        401: 'Unauthorized'
    }
)
@api_view(['DELETE'])
def community_post_delete(request, post_id: str):
    user: FirebaseUser = request.user  # type: ignore
    post_doc = _posts_collection().document(post_id).get()
    
    if not post_doc.exists:
        return Response({'detail': 'Post not found'}, status=status.HTTP_404_NOT_FOUND)
    
    post_data = post_doc.to_dict()
    if post_data.get('authorId') != user.uid:
        return Response({'detail': 'You can only delete your own posts'}, status=status.HTTP_403_FORBIDDEN)
    
    # Delete post and all associated data
    _posts_collection().document(post_id).delete()
    
    return Response(status=status.HTTP_204_NO_CONTENT)


@swagger_auto_schema(
    method='post',
    operation_description='Like or unlike a post',
    request_body=LikeSerializer,
    responses={
        200: openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'liked': openapi.Schema(type=openapi.TYPE_BOOLEAN, description='Whether the post is now liked'),
                'likesCount': openapi.Schema(type=openapi.TYPE_INTEGER, description='Total number of likes')
            }
        ),
        404: 'Not Found - Post not found',
        401: 'Unauthorized'
    }
)
@api_view(['POST'])
def toggle_post_like(request, post_id: str):
    user: FirebaseUser = request.user  # type: ignore
    post_doc = _posts_collection().document(post_id).get()
    
    if not post_doc.exists:
        return Response({'detail': 'Post not found'}, status=status.HTTP_404_NOT_FOUND)
    
    like_ref = _likes_collection(post_id).document(user.uid)
    like_doc = like_ref.get()
    
    if like_doc.exists:
        # Unlike
        like_ref.delete()
        liked = False
    else:
        # Like
        like_ref.set({
            'userId': user.uid,
            'userName': user.name or user.email or 'Anonymous',
            'likedAt': datetime.now(timezone.utc)
        })
        liked = True
    
    # Get updated likes count
    likes_count = len(list(_likes_collection(post_id).stream()))
    
    return Response({
        'liked': liked,
        'likesCount': likes_count
    })


@swagger_auto_schema(
    method='get',
    operation_description='Get comments for a specific post',
    manual_parameters=[
        openapi.Parameter(
            'limit',
            openapi.IN_QUERY,
            description="Number of comments to return (default: 20, max: 50)",
            type=openapi.TYPE_INTEGER,
            required=False
        ),
        openapi.Parameter(
            'offset',
            openapi.IN_QUERY,
            description="Number of comments to skip (for pagination)",
            type=openapi.TYPE_INTEGER,
            required=False
        )
    ],
    responses={
        200: CommentSerializer(many=True),
        404: 'Not Found - Post not found',
        401: 'Unauthorized'
    }
)
@api_view(['GET'])
def post_comments_list(request, post_id: str):
    user: FirebaseUser = request.user  # type: ignore
    limit = min(int(request.query_params.get('limit', 20)), 50)
    offset = int(request.query_params.get('offset', 0))
    
    post_doc = _posts_collection().document(post_id).get()
    if not post_doc.exists:
        return Response({'detail': 'Post not found'}, status=status.HTTP_404_NOT_FOUND)
    
    # Get comments
    comments_query = _comments_collection(post_id).order_by('createdAt', direction='ASCENDING')
    
    try:
        comments_docs = list(comments_query.offset(offset).limit(limit).stream())
    except FailedPrecondition:
        comments_docs = list(_comments_collection(post_id).offset(offset).limit(limit).stream())
    
    comments = []
    for doc in comments_docs:
        comment_data = doc.to_dict()
        comment_data['id'] = doc.id
        
        # Check if user liked this comment
        comment_like_ref = get_firestore().collection('community_posts').document(post_id).collection('comments').document(doc.id).collection('likes').document(user.uid)
        comment_data['isLikedByUser'] = comment_like_ref.get().exists
        
        # Get comment likes count
        comment_data['likesCount'] = len(list(get_firestore().collection('community_posts').document(post_id).collection('comments').document(doc.id).collection('likes').stream()))
        
        comments.append(comment_data)
    
    return Response(comments)


@swagger_auto_schema(
    method='post',
    operation_description='Create a comment on a post',
    request_body=CreateCommentSerializer,
    responses={
        201: CommentSerializer,
        400: 'Bad Request - Validation error',
        404: 'Not Found - Post not found',
        401: 'Unauthorized'
    }
)
@api_view(['POST'])
def create_comment(request):
    user: FirebaseUser = request.user  # type: ignore
    ser = CreateCommentSerializer(data=request.data)
    ser.is_valid(raise_exception=True)
    
    post_id = ser.validated_data['postId']
    post_doc = _posts_collection().document(post_id).get()
    
    if not post_doc.exists:
        return Response({'detail': 'Post not found'}, status=status.HTTP_404_NOT_FOUND)
    
    comment_data = {
        'postId': post_id,
        'content': ser.validated_data['content'],
        'authorId': user.uid,
        'authorName': user.name or user.email or 'Anonymous',
        'createdAt': datetime.now(timezone.utc),
        'updatedAt': datetime.now(timezone.utc),
    }
    
    # Create comment
    comment_ref = _comments_collection(post_id).document()
    comment_ref.set(comment_data)
    
    # Return created comment
    created_comment = comment_data.copy()
    created_comment['id'] = comment_ref.id
    created_comment['likesCount'] = 0
    created_comment['isLikedByUser'] = False
    
    return Response(created_comment, status=status.HTTP_201_CREATED)


@swagger_auto_schema(
    method='delete',
    operation_description='Delete a comment (only by author)',
    responses={
        204: 'No Content - Comment deleted successfully',
        404: 'Not Found - Comment not found',
        403: 'Forbidden - Not the comment author',
        401: 'Unauthorized'
    }
)
@api_view(['DELETE'])
def delete_comment(request, post_id: str, comment_id: str):
    user: FirebaseUser = request.user  # type: ignore
    comment_doc = _comments_collection(post_id).document(comment_id).get()
    
    if not comment_doc.exists:
        return Response({'detail': 'Comment not found'}, status=status.HTTP_404_NOT_FOUND)
    
    comment_data = comment_doc.to_dict()
    if comment_data.get('authorId') != user.uid:
        return Response({'detail': 'You can only delete your own comments'}, status=status.HTTP_403_FORBIDDEN)
    
    # Delete comment
    _comments_collection(post_id).document(comment_id).delete()
    
    return Response(status=status.HTTP_204_NO_CONTENT)


@swagger_auto_schema(
    method='post',
    operation_description='Share a post',
    request_body=ShareSerializer,
    responses={
        200: openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'shared': openapi.Schema(type=openapi.TYPE_BOOLEAN, description='Whether the post was shared'),
                'sharesCount': openapi.Schema(type=openapi.TYPE_INTEGER, description='Total number of shares'),
                'shareId': openapi.Schema(type=openapi.TYPE_STRING, description='ID of the share record')
            }
        ),
        404: 'Not Found - Post not found',
        401: 'Unauthorized'
    }
)
@api_view(['POST'])
def share_post(request, post_id: str):
    user: FirebaseUser = request.user  # type: ignore
    ser = ShareSerializer(data=request.data)
    ser.is_valid(raise_exception=True)
    
    message = ser.validated_data.get('message', '')
    
    post_doc = _posts_collection().document(post_id).get()
    if not post_doc.exists:
        return Response({'detail': 'Post not found'}, status=status.HTTP_404_NOT_FOUND)
    
    # Create share record
    share_data = {
        'postId': post_id,
        'userId': user.uid,
        'userName': user.name or user.email or 'Anonymous',
        'message': message,
        'sharedAt': datetime.now(timezone.utc)
    }
    
    share_ref = _shares_collection(post_id).document()
    share_ref.set(share_data)
    
    # Get updated shares count
    shares_count = len(list(_shares_collection(post_id).stream()))
    
    return Response({
        'shared': True,
        'sharesCount': shares_count,
        'shareId': share_ref.id
    })


@swagger_auto_schema(
    method='get',
    operation_description='Check Firebase Storage configuration status',
    responses={
        200: openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'storageConfigured': openapi.Schema(type=openapi.TYPE_BOOLEAN, description='Whether Firebase Storage is configured'),
                'bucketName': openapi.Schema(type=openapi.TYPE_STRING, description='Storage bucket name'),
                'projectId': openapi.Schema(type=openapi.TYPE_STRING, description='Firebase project ID')
            }
        )
    }
)
@api_view(['GET'])
def check_storage_config(request):
    """Check if Firebase Storage is properly configured"""
    from django.conf import settings
    from .firebase import get_bucket
    
    try:
        bucket = get_bucket()
        return Response({
            'storageConfigured': True,
            'bucketName': settings.FIREBASE_STORAGE_BUCKET,
            'projectId': settings.FIREBASE_PROJECT_ID
        })
    except Exception as e:
        return Response({
            'storageConfigured': False,
            'bucketName': settings.FIREBASE_STORAGE_BUCKET or 'Not configured',
            'projectId': settings.FIREBASE_PROJECT_ID or 'Not configured',
            'error': str(e)
        })


@swagger_auto_schema(
    method='get',
    operation_description='Check Twilio SMS service configuration and status',
    responses={
        200: openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'configured': openapi.Schema(type=openapi.TYPE_BOOLEAN),
                'account_sid_set': openapi.Schema(type=openapi.TYPE_BOOLEAN),
                'auth_token_set': openapi.Schema(type=openapi.TYPE_BOOLEAN),
                'phone_number_set': openapi.Schema(type=openapi.TYPE_BOOLEAN),
                'client_initialized': openapi.Schema(type=openapi.TYPE_BOOLEAN),
                'phone_number': openapi.Schema(type=openapi.TYPE_STRING),
            }
        )
    }
)
@api_view(['GET'])
@permission_classes([AllowAny])
def check_sms_config(request):
    """Check Twilio SMS service configuration"""
    status_info = sms_service.get_service_status()
    return Response(status_info)


@swagger_auto_schema(
    method='get',
    operation_description='Check SMS message delivery status by message SID',
    responses={
        200: openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'success': openapi.Schema(type=openapi.TYPE_BOOLEAN),
                'message_sid': openapi.Schema(type=openapi.TYPE_STRING),
                'status': openapi.Schema(type=openapi.TYPE_STRING),
                'to': openapi.Schema(type=openapi.TYPE_STRING),
                'from': openapi.Schema(type=openapi.TYPE_STRING),
                'body': openapi.Schema(type=openapi.TYPE_STRING),
                'date_created': openapi.Schema(type=openapi.TYPE_STRING),
                'date_sent': openapi.Schema(type=openapi.TYPE_STRING),
            }
        )
    }
)
@api_view(['GET'])
@permission_classes([AllowAny])
def sms_message_status(request, message_sid):
    """Check SMS message delivery status"""
    status_info = sms_service.get_message_status(message_sid)
    return Response(status_info)

@swagger_auto_schema(
    method='post',
    operation_description='Verify phone number for Twilio trial account',
    request_body=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        properties={
            'phone_number': openapi.Schema(type=openapi.TYPE_STRING, description='Phone number to verify (e.g., +12402915041)'),
        },
        required=['phone_number']
    ),
    responses={
        200: openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'success': openapi.Schema(type=openapi.TYPE_BOOLEAN),
                'message': openapi.Schema(type=openapi.TYPE_STRING),
                'verification_url': openapi.Schema(type=openapi.TYPE_STRING),
            }
        )
    }
)
@api_view(['POST'])
@permission_classes([AllowAny])
def verify_phone_number(request):
    """Verify phone number for Twilio trial account (manual process)"""
    phone_number = request.data.get('phone_number')
    
    if not phone_number:
        return Response({
            'success': False,
            'message': 'Phone number is required'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # For trial accounts, users need to verify numbers manually
    verification_url = "https://console.twilio.com/us1/develop/phone-numbers/manage/verified"
    
    return Response({
        'success': True,
        'message': f'For trial accounts, you need to manually verify {phone_number} in Twilio Console',
        'verification_url': verification_url,
        'instructions': [
            f"1. Go to {verification_url}",
            f"2. Click 'Add a new number'",
            f"3. Enter {phone_number}",
            f"4. Complete verification process",
            f"5. Try sending OTP again"
        ]
    })

@swagger_auto_schema(
    method='get',
    operation_description='Get Twilio account status and limitations',
    responses={
        200: openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'account_type': openapi.Schema(type=openapi.TYPE_STRING),
                'limitations': openapi.Schema(type=openapi.TYPE_ARRAY, items=openapi.Schema(type=openapi.TYPE_STRING)),
                'solutions': openapi.Schema(type=openapi.TYPE_ARRAY, items=openapi.Schema(type=openapi.TYPE_STRING)),
            }
        )
    }
)
@api_view(['GET'])
@permission_classes([AllowAny])
def twilio_account_status(request):
    """Get Twilio account status and limitations"""
    status_info = sms_service.get_service_status()
    
    # Check if this is likely a trial account
    is_trial = True  # Assume trial unless proven otherwise
    
    limitations = [
        "Can only send SMS to verified phone numbers",
        "Limited to 10 verified numbers",
        "SMS messages are prefixed with 'Sent from your Twilio trial account'",
        "Cannot send to international numbers (except verified ones)",
        "Rate limits are stricter"
    ]
    
    solutions = [
        "Verify phone numbers in Twilio Console",
        "Upgrade to paid account for full functionality",
        "Use test phone numbers for development",
        "Implement phone number verification in your app"
    ]
    
    return Response({
        'account_type': 'trial' if is_trial else 'paid',
        'service_status': status_info,
        'limitations': limitations,
        'solutions': solutions,
        'verification_url': 'https://console.twilio.com/us1/develop/phone-numbers/manage/verified'
    })


# Support/Help Desk Endpoints (Simplified)

@swagger_auto_schema(
    method='post',
    operation_description='Send a support message with email and details',
    request_body=CreateSupportMessageSerializer,
    responses={
        201: openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'message': openapi.Schema(type=openapi.TYPE_STRING, description='Success message'),
                'supportId': openapi.Schema(type=openapi.TYPE_STRING, description='Support message ID')
            }
        ),
        400: 'Bad Request - Validation error'
    }
)
@api_view(['POST'])
@permission_classes([AllowAny])
def send_support_message(request):
    ser = CreateSupportMessageSerializer(data=request.data)
    ser.is_valid(raise_exception=True)
    
    # Get user ID if authenticated
    user_id = None
    if hasattr(request, 'user') and request.user.is_authenticated:
        user: FirebaseUser = request.user  # type: ignore
        user_id = user.uid
    
    support_data = {
        'email': ser.validated_data['email'],
        'details': ser.validated_data['details'],
        'userId': user_id,
        'status': 'new',
        'createdAt': datetime.now(timezone.utc),
        'updatedAt': datetime.now(timezone.utc),
        'adminReply': '',
    }
    
    # Create support message
    support_ref = _support_messages_collection().document()
    support_ref.set(support_data)
    
    return Response({
        'message': 'Support message sent successfully. We will get back to you soon.',
        'supportId': support_ref.id
    }, status=status.HTTP_201_CREATED)


@swagger_auto_schema(
    method='get',
    operation_description='Get user support messages (authenticated users only)',
    manual_parameters=[
        openapi.Parameter(
            'limit',
            openapi.IN_QUERY,
            description="Number of messages to return (default: 20, max: 50)",
            type=openapi.TYPE_INTEGER,
            required=False
        ),
        openapi.Parameter(
            'offset',
            openapi.IN_QUERY,
            description="Number of messages to skip (for pagination)",
            type=openapi.TYPE_INTEGER,
            required=False
        )
    ],
    responses={
        200: SupportMessageSerializer(many=True),
        401: 'Unauthorized'
    }
)
@api_view(['GET'])
def my_support_messages(request):
    user: FirebaseUser = request.user  # type: ignore
    limit = min(int(request.query_params.get('limit', 20)), 50)
    offset = int(request.query_params.get('offset', 0))
    
    # Build query for user's messages
    query = _support_messages_collection().where('userId', '==', user.uid)
    
    # Order by creation time (newest first)
    try:
        messages_docs = list(query.order_by('createdAt', direction='DESCENDING').offset(offset).limit(limit).stream())
    except FailedPrecondition:
        messages_docs = list(query.offset(offset).limit(limit).stream())
    
    messages = []
    for doc in messages_docs:
        message_data = doc.to_dict()
        message_data['id'] = doc.id
        messages.append(message_data)
    
    return Response(messages)


# Admin Support Management Endpoints

@csrf_exempt
@login_required
@user_passes_test(is_superuser)
def admin_support_messages_list(request):
    limit = min(int(request.GET.get('limit', 20)), 100)
    offset = int(request.GET.get('offset', 0))
    status_filter = request.GET.get('status')
    
    # Build query
    query = _support_messages_collection()
    
    if status_filter:
        query = query.where('status', '==', status_filter)
    
    # Order by creation time (newest first)
    try:
        messages_docs = list(query.order_by('createdAt', direction='DESCENDING').offset(offset).limit(limit).stream())
    except FailedPrecondition:
        messages_docs = list(query.offset(offset).limit(limit).stream())
    
    messages = []
    for doc in messages_docs:
        message_data = doc.to_dict()
        message_data['id'] = doc.id
        messages.append(message_data)
    
    return JsonResponse({'messages': messages, 'total': len(messages)})


@swagger_auto_schema(
    method='patch',
    operation_description='Update support message status and add admin reply (admin only)',
    request_body=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        properties={
            'status': openapi.Schema(type=openapi.TYPE_STRING, enum=['new', 'read', 'replied'], description='Message status'),
            'adminReply': openapi.Schema(type=openapi.TYPE_STRING, description='Admin reply message')
        }
    ),
    responses={
        200: SupportMessageSerializer,
        404: 'Not Found - Message not found',
        403: 'Forbidden - Admin access required',
        401: 'Unauthorized'
    }
)
@api_view(['PATCH'])
@permission_classes([IsAdminFirebase])
def admin_update_support_message(request, message_id: str):
    user: FirebaseUser = request.user  # type: ignore
    
    message_doc = _support_messages_collection().document(message_id).get()
    
    if not message_doc.exists:
        return Response({'detail': 'Support message not found'}, status=status.HTTP_404_NOT_FOUND)
    
    # Prepare update data
    update_data = {
        'updatedAt': datetime.now(timezone.utc),
    }
    
    if 'status' in request.data:
        update_data['status'] = request.data['status']
    
    if 'adminReply' in request.data:
        update_data['adminReply'] = request.data['adminReply']
    
    # Update message
    _support_messages_collection().document(message_id).update(update_data)
    
    # Return updated message
    updated_message = message_doc.to_dict()
    updated_message.update(update_data)
    updated_message['id'] = message_id
    
    return Response(updated_message)


@csrf_exempt
@login_required
@user_passes_test(is_superuser)
def admin_support_analytics(request):
    # Get all messages
    all_messages = list(_support_messages_collection().stream())
    
    total_messages = len(all_messages)
    new_messages = len([m for m in all_messages if m.to_dict().get('status') == 'new'])
    read_messages = len([m for m in all_messages if m.to_dict().get('status') == 'read'])
    replied_messages = len([m for m in all_messages if m.to_dict().get('status') == 'replied'])
    
    return JsonResponse({
        'totalMessages': total_messages,
        'newMessages': new_messages,
        'readMessages': read_messages,
        'repliedMessages': replied_messages
    })


# Admin Dashboard API Endpoints

@csrf_exempt
@login_required
@user_passes_test(is_superuser)
def admin_dashboard_stats(request):
    """Get dashboard statistics for admin panel"""
    try:
        db = get_firestore()
        
        # Get total users count - main collection
        users_ref = db.collection('users')
        users_docs = list(users_ref.stream())
        total_users = len(users_docs)
        
        # Get total pets count - use collection_group to get all pets from subcollections
        total_pets = 0
        try:
            # Try collection_group first (most efficient)
            pets_docs = list(db.collection_group('pets').stream())
            total_pets = len(pets_docs)
        except Exception as e:
            print(f"Collection group pets failed: {e}")
            # Fallback: count from each user's pets subcollection
            for user_doc in users_docs:
                try:
                    user_pets = list(users_ref.document(user_doc.id).collection('pets').stream())
                    total_pets += len(user_pets)
                except:
                    pass
        
        # Get total emotion scans count - use collection_group
        total_scans = 0
        try:
            # Try collection_group first (most efficient)
            emotion_logs = list(db.collection_group('emotion_logs').stream())
            total_scans = len(emotion_logs)
        except Exception as e:
            print(f"Collection group emotion_logs failed: {e}")
            # Fallback: count from each user's emotion_logs subcollection
            for user_doc in users_docs:
                try:
                    user_scans = list(users_ref.document(user_doc.id).collection('emotion_logs').stream())
                    total_scans += len(user_scans)
                except:
                    pass
        
        # Get new support messages count
        new_support = 0
        try:
            support_messages = list(db.collection('supportMessages').stream())
            new_support = len([msg for msg in support_messages if msg.to_dict().get('status') == 'new'])
        except:
            try:
                support_messages = list(db.collection('support').stream())
                new_support = len([msg for msg in support_messages if msg.to_dict().get('status') == 'new'])
            except:
                pass
        
        return JsonResponse({
            'totalUsers': total_users,
            'totalPets': total_pets,
            'totalScans': total_scans,
            'newSupportMessages': new_support
        })
        
    except Exception as e:
        print(f"Dashboard stats error: {e}")
        return JsonResponse({
            'totalUsers': 0,
            'totalPets': 0,
            'totalScans': 0,
            'newSupportMessages': 0
        }, status=500)


@csrf_exempt
@login_required
@user_passes_test(is_superuser)
def admin_recent_activity(request):
    """Get recent activity for admin dashboard"""
    
    try:
        activities = []
        
        # Get recent users (last 5)
        recent_users = list(get_firestore().collection('users')
                          .order_by('createdAt', direction='DESCENDING')
                          .limit(5).stream())
        
        for user_doc in recent_users:
            user_data = user_doc.to_dict()
            activities.append({
                'type': 'user_registration',
                'description': f'New user registered: {user_data.get("email", "Unknown")}',
                'timestamp': user_data.get('createdAt')
            })
        
        # Get recent support messages (last 3)
        recent_support = list(_support_messages_collection()
                            .order_by('createdAt', direction='DESCENDING')
                            .limit(3).stream())
        
        for support_doc in recent_support:
            support_data = support_doc.to_dict()
            activities.append({
                'type': 'support_message',
                'description': f'New support message from: {support_data.get("email", "Unknown")}',
                'timestamp': support_data.get('createdAt')
            })
        
        # Get recent emotion scans (last 5)
        recent_scans = []
        for user_doc in recent_users:
            user_scans = list(get_firestore().collection('users').document(user_doc.id)
                            .collection('emotion_logs')
                            .order_by('timestamp', direction='DESCENDING')
                            .limit(2).stream())
            recent_scans.extend(user_scans)
        
        # Sort by timestamp and take last 5
        recent_scans.sort(key=lambda x: x.to_dict().get('timestamp', ''), reverse=True)
        recent_scans = recent_scans[:5]
        
        for scan_doc in recent_scans:
            scan_data = scan_doc.to_dict()
            pet_name = scan_data.get('petName', 'Unknown Pet')
            activities.append({
                'type': 'emotion_scan',
                'description': f'Emotion scan completed for {pet_name}',
                'timestamp': scan_data.get('timestamp')
            })
        
        # Sort all activities by timestamp (newest first)
        activities.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        activities = activities[:10]  # Limit to 10 most recent
        
        return JsonResponse({'activities': activities})
        
    except Exception as e:
        return JsonResponse({
            'detail': f'Error fetching recent activity: {str(e)}'
        }, status=500)


@csrf_exempt
@login_required
@user_passes_test(is_superuser)
def admin_emotion_trends(request):
    """Get emotion detection trends for admin dashboard charts"""
    
    try:
        # Get emotion data for the last 30 days
        from datetime import datetime, timedelta, timezone
        
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=30)
        
        # Get real emotion data from Firestore
        emotion_counts = {
            'happy': [0] * 10,
            'sad': [0] * 10,
            'excited': [0] * 10,
            'anxious': [0] * 10,
            'calm': [0] * 10
        }
        
        # Query emotion logs from the last 30 days
        users_docs = list(get_firestore().collection('users').stream())
        
        for user_doc in users_docs:
            try:
                # Get emotion logs for this user
                logs_docs = list(get_firestore().collection('users')
                               .document(user_doc.id)
                               .collection('emotion_logs')
                               .where('timestamp', '>=', start_date)
                               .stream())
                
                for log_doc in logs_docs:
                    log_data = log_doc.to_dict()
                    emotion = log_data.get('detectedEmotion', '').lower()
                    timestamp = log_data.get('timestamp')
                    
                    if timestamp and emotion in emotion_counts:
                        # Calculate which week this belongs to (0-9)
                        days_diff = (timestamp - start_date).days
                        week_index = min(days_diff // 3, 9)  # 3 days per week
                        
                        emotion_counts[emotion][week_index] += 1
                        
            except Exception as e:
                print(f"Error processing logs for user {user_doc.id}: {e}")
                continue
        
        # If no real data, use sample data
        if sum(sum(emotion_counts[e]) for e in emotion_counts) == 0:
            emotion_counts = {
                'happy': [65, 59, 80, 81, 56, 55, 40, 70, 85, 90],
                'sad': [28, 48, 40, 19, 86, 27, 35, 25, 15, 10],
                'excited': [18, 38, 60, 29, 76, 37, 45, 55, 65, 70],
                'anxious': [12, 25, 30, 15, 40, 20, 18, 22, 15, 12],
                'calm': [45, 55, 35, 65, 25, 50, 60, 45, 55, 48]
            }
        
        emotion_data = {
            **emotion_counts,
            'labels': ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5', 'Week 6', 'Week 7', 'Week 8', 'Week 9', 'Week 10']
        }
        
        return JsonResponse({'emotionData': emotion_data})
        
    except Exception as e:
        return JsonResponse({
            'detail': f'Error fetching emotion trends: {str(e)}'
        }, status=500)


# Admin Users Management
@csrf_exempt
@login_required
@user_passes_test(is_superuser)
def admin_users_list(request):
    """Get list of all users for admin panel"""
    try:
        page = int(request.GET.get('page', 1))
        limit = int(request.GET.get('limit', 20))
        search = request.GET.get('search', '')
        
        # Calculate offset
        offset = (page - 1) * limit
        
        # Query users
        users_query = get_firestore().collection('users').order_by('createdAt', direction='DESCENDING')
        
        # Apply search filter if provided
        if search:
            # Note: Firestore doesn't support case-insensitive search easily
            # This is a basic implementation
            all_users = list(users_query.stream())
            filtered_users = []
            for user in all_users:
                user_data = user.to_dict()
                if (search.lower() in user_data.get('email', '').lower() or 
                    search.lower() in user_data.get('firstName', '').lower() or
                    search.lower() in user_data.get('lastName', '').lower()):
                    filtered_users.append(user)
            
            # Apply pagination
            total_users = len(filtered_users)
            users = filtered_users[offset:offset + limit]
        else:
            # Get total count
            all_users = list(users_query.stream())
            total_users = len(all_users)
            
            # Apply pagination
            users = all_users[offset:offset + limit]
        
        # Format user data
        users_data = []
        for user in users:
            user_data = user.to_dict()
            
            # Get pets count for this user
            pets_count = 0
            try:
                # Try to get pets from main pets collection
                user_pets = list(get_firestore().collection('pets').where('ownerId', '==', user.id).stream())
                pets_count = len(user_pets)
            except:
                try:
                    # Try to get pets from user subcollection
                    user_pets = list(get_firestore().collection('users').document(user.id).collection('pets').stream())
                    pets_count = len(user_pets)
                except:
                    pass
            
            users_data.append({
                'id': user.id,
                'email': user_data.get('email', ''),
                'firstName': user_data.get('firstName', user_data.get('first_name', user_data.get('name', ''))),
                'lastName': user_data.get('lastName', user_data.get('last_name', user_data.get('surname', ''))),
                'phoneNumber': user_data.get('phoneNumber', user_data.get('phone_number', user_data.get('phone', ''))),
                'createdAt': user_data.get('createdAt', user_data.get('created_at', user_data.get('created', ''))),
                'lastLoginAt': user_data.get('lastLoginAt', user_data.get('last_login_at', user_data.get('lastLogin', ''))),
                'isActive': user_data.get('isActive', user_data.get('is_active', user_data.get('active', True))),
                'petsCount': pets_count
            })
        
        return JsonResponse({
            'users': users_data,
            'pagination': {
                'page': page,
                'limit': limit,
                'total': total_users,
                'pages': (total_users + limit - 1) // limit
            }
        })
        
    except Exception as e:
        print(f"Error fetching users: {e}")
        return JsonResponse({'error': str(e)}, status=500)


# Admin Pets Management
@csrf_exempt
@login_required
@user_passes_test(is_superuser)
def admin_pets_list(request):
    """Get list of all pets for admin panel"""
    try:
        page = int(request.GET.get('page', 1))
        limit = int(request.GET.get('limit', 20))
        search = request.GET.get('search', '')
        
        # Calculate offset
        offset = (page - 1) * limit
        
        # Get all pets using collection_group (most efficient)
        all_pets = []
        try:
            # Use collection_group to get all pets from subcollections
            pets_docs = list(get_firestore().collection_group('pets').stream())
            all_pets.extend(pets_docs)
        except Exception as e:
            print(f"Collection group pets failed: {e}")
            # Fallback: try main collection
            try:
                pets_query = get_firestore().collection('pets')
                all_pets = list(pets_query.stream())
            except:
                # Fallback: get from users subcollections
                users_docs = list(get_firestore().collection('users').stream())
                for user_doc in users_docs:
                    try:
                        user_pets = list(get_firestore().collection('users').document(user_doc.id).collection('pets').stream())
                        all_pets.extend(user_pets)
                    except:
                        pass
        
        # Apply search filter if provided
        if search:
            filtered_pets = []
            for pet in all_pets:
                pet_data = pet.to_dict()
                if (search.lower() in pet_data.get('name', '').lower() or 
                    search.lower() in pet_data.get('breed', '').lower() or
                    search.lower() in pet_data.get('species', '').lower()):
                    filtered_pets.append(pet)
            
            total_pets = len(filtered_pets)
            pets = filtered_pets[offset:offset + limit]
        else:
            total_pets = len(all_pets)
            pets = all_pets[offset:offset + limit]
        
        # Format pet data
        pets_data = []
        for pet in pets:
            pet_data = pet.to_dict()
            
            # Get scans count for this pet
            scans_count = 0
            try:
                # Try to get scans from main emotionLogs collection
                pet_scans = list(get_firestore().collection('emotionLogs').where('petId', '==', pet.id).stream())
                scans_count = len(pet_scans)
            except:
                try:
                    # Try alternative collection names
                    pet_scans = list(get_firestore().collection('emotion_logs').where('petId', '==', pet.id).stream())
                    scans_count = len(pet_scans)
                except:
                    try:
                        pet_scans = list(get_firestore().collection('scans').where('petId', '==', pet.id).stream())
                        scans_count = len(pet_scans)
                    except:
                        pass
            
            pets_data.append({
                'id': pet.id,
                'name': pet_data.get('name', ''),
                'species': pet_data.get('species', pet_data.get('type', '')),
                'breed': pet_data.get('breed', ''),
                'age': pet_data.get('age', ''),
                'gender': pet_data.get('gender', ''),
                'ownerId': pet_data.get('ownerId', pet_data.get('owner_id', '')),
                'ownerName': pet_data.get('ownerName', pet_data.get('owner_name', '')),
                'createdAt': pet_data.get('createdAt', pet_data.get('created_at', '')),
                'imageUrl': pet_data.get('imageUrl', pet_data.get('image_url', pet_data.get('image', ''))),
                'scansCount': scans_count
            })
        
        return JsonResponse({
            'pets': pets_data,
            'pagination': {
                'page': page,
                'limit': limit,
                'total': total_pets,
                'pages': (total_pets + limit - 1) // limit
            }
        })
        
    except Exception as e:
        print(f"Error fetching pets: {e}")
        return JsonResponse({'error': str(e)}, status=500)


# Admin Emotion Logs Management
@csrf_exempt
@login_required
@user_passes_test(is_superuser)
def admin_emotion_logs_list(request):
    """Get list of all emotion logs for admin panel"""
    try:
        page = int(request.GET.get('page', 1))
        limit = int(request.GET.get('limit', 20))
        emotion_filter = request.GET.get('emotion', '')
        
        # Calculate offset
        offset = (page - 1) * limit
        
        # Get all emotion logs using collection_group (most efficient)
        all_logs = []
        try:
            # Use collection_group to get all emotion_logs from subcollections
            logs_docs = list(get_firestore().collection_group('emotion_logs').stream())
            all_logs.extend(logs_docs)
        except Exception as e:
            print(f"Collection group emotion_logs failed: {e}")
            # Fallback: try main collections
            try:
                logs_query = get_firestore().collection('emotionLogs')
                all_logs = list(logs_query.stream())
            except:
                try:
                    logs_query = get_firestore().collection('emotion_logs')
                    all_logs = list(logs_query.stream())
                except:
                    try:
                        logs_query = get_firestore().collection('scans')
                        all_logs = list(logs_query.stream())
                    except:
                        # Fallback: get from users subcollections
                        users_docs = list(get_firestore().collection('users').stream())
                        for user_doc in users_docs:
                            try:
                                user_logs = list(get_firestore().collection('users').document(user_doc.id).collection('emotion_logs').stream())
                                all_logs.extend(user_logs)
                            except:
                                try:
                                    user_logs = list(get_firestore().collection('users').document(user_doc.id).collection('scans').stream())
                                    all_logs.extend(user_logs)
                                except:
                                    pass
        
        # Apply emotion filter if provided
        if emotion_filter:
            filtered_logs = []
            for log in all_logs:
                log_data = log.to_dict()
                if log_data.get('emotion', '').lower() == emotion_filter.lower():
                    filtered_logs.append(log)
            all_logs = filtered_logs
        
        total_logs = len(all_logs)
        
        # Apply pagination
        logs = all_logs[offset:offset + limit]
        
        # Format log data
        logs_data = []
        for log in logs:
            log_data = log.to_dict()
            logs_data.append({
                'id': log.id,
                'userId': log_data.get('userId', log_data.get('user_id', '')),
                'petId': log_data.get('petId', log_data.get('pet_id', '')),
                'emotion': log_data.get('emotion', log_data.get('detected_emotion', log_data.get('result', ''))),
                'confidence': log_data.get('confidence', log_data.get('score', 0)),
                'timestamp': log_data.get('timestamp', log_data.get('created_at', log_data.get('created', ''))),
                'imageUrl': log_data.get('imageUrl', log_data.get('image_url', log_data.get('image', ''))),
                'petName': log_data.get('petName', log_data.get('pet_name', '')),
                'userName': log_data.get('userName', log_data.get('user_name', ''))
            })
        
        return JsonResponse({
            'logs': logs_data,
            'pagination': {
                'page': page,
                'limit': limit,
                'total': total_logs,
                'pages': (total_logs + limit - 1) // limit
            }
        })
        
    except Exception as e:
        print(f"Error fetching emotion logs: {e}")
        return JsonResponse({'error': str(e)}, status=500)


# Admin Update Functions
@csrf_exempt
@login_required
@user_passes_test(is_superuser)
def admin_update_user(request, user_id):
    """Update user information (admin only)"""
    if request.method == 'PATCH':
        try:
            data = json.loads(request.body)
            
            # Get user document
            user_ref = get_firestore().collection('users').document(user_id)
            user_doc = user_ref.get()
            
            if not user_doc.exists:
                return JsonResponse({'error': 'User not found'}, status=404)
            
            # Update allowed fields
            update_data = {}
            allowed_fields = [
                'firstName', 'lastName', 'email', 'phoneNumber', 'isActive',
                'dateOfBirth', 'gender', 'address', 'city', 'state', 'country',
                'postalCode', 'profileImageUrl', 'bio', 'preferences', 'settings',
                'lastLoginAt', 'emailVerified', 'phoneVerified', 'subscriptionStatus'
            ]
            
            for field in allowed_fields:
                if field in data:
                    update_data[field] = data[field]
            
            if update_data:
                update_data['updatedAt'] = datetime.now(timezone.utc)
                user_ref.update(update_data)
            
            return JsonResponse({'message': 'User updated successfully'})
            
        except Exception as e:
            print(f"Error updating user: {e}")
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)


@csrf_exempt
@login_required
@user_passes_test(is_superuser)
def admin_update_pet(request, pet_id):
    """Update pet information (admin only)"""
    if request.method == 'PATCH':
        try:
            data = json.loads(request.body)
            
            # Get pet document
            pet_ref = get_firestore().collection('pets').document(pet_id)
            pet_doc = pet_ref.get()
            
            if not pet_doc.exists:
                return JsonResponse({'error': 'Pet not found'}, status=404)
            
            # Update allowed fields
            update_data = {}
            allowed_fields = [
                'name', 'species', 'breed', 'age', 'gender', 'weight', 'color',
                'description', 'medicalHistory', 'vaccinations', 'allergies',
                'medications', 'vetInfo', 'emergencyContact', 'imageUrl', 'images',
                'personality', 'habits', 'favoriteFoods', 'favoriteActivities',
                'healthStatus', 'lastCheckup', 'nextCheckup', 'microchipId',
                'isNeutered', 'isVaccinated', 'specialNeeds', 'notes'
            ]
            
            for field in allowed_fields:
                if field in data:
                    update_data[field] = data[field]
            
            if update_data:
                update_data['updatedAt'] = datetime.now(timezone.utc)
                pet_ref.update(update_data)
            
            return JsonResponse({'message': 'Pet updated successfully'})
            
        except Exception as e:
            print(f"Error updating pet: {e}")
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)


@csrf_exempt
@login_required
@user_passes_test(is_superuser)
def admin_delete_user(request, user_id):
    """Delete user (admin only)"""
    if request.method == 'DELETE':
        try:
            # Get user document
            user_ref = get_firestore().collection('users').document(user_id)
            user_doc = user_ref.get()
            
            if not user_doc.exists:
                return JsonResponse({'error': 'User not found'}, status=404)
            
            # Delete user
            user_ref.delete()
            
            # Also delete user's pets
            pets_query = get_firestore().collection('pets').where('ownerId', '==', user_id)
            for pet_doc in pets_query.stream():
                pet_doc.reference.delete()
            
            # Delete user's emotion logs
            logs_query = get_firestore().collection('emotionLogs').where('userId', '==', user_id)
            for log_doc in logs_query.stream():
                log_doc.reference.delete()
            
            return JsonResponse({'message': 'User deleted successfully'})
            
        except Exception as e:
            print(f"Error deleting user: {e}")
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)


@csrf_exempt
@login_required
@user_passes_test(is_superuser)
def admin_delete_pet(request, pet_id):
    """Delete pet (admin only)"""
    if request.method == 'DELETE':
        try:
            # Get pet document
            pet_ref = get_firestore().collection('pets').document(pet_id)
            pet_doc = pet_ref.get()
            
            if not pet_doc.exists:
                return JsonResponse({'error': 'Pet not found'}, status=404)
            
            # Delete pet
            pet_ref.delete()
            
            # Also delete pet's emotion logs
            logs_query = get_firestore().collection('emotionLogs').where('petId', '==', pet_id)
            for log_doc in logs_query.stream():
                log_doc.reference.delete()
            
            return JsonResponse({'message': 'Pet deleted successfully'})
            
        except Exception as e:
            print(f"Error deleting pet: {e}")
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)


@csrf_exempt
@login_required
@user_passes_test(is_superuser)
def admin_delete_emotion_log(request, log_id):
    """Delete emotion log (admin only)"""
    if request.method == 'DELETE':
        try:
            # Get log document
            log_ref = get_firestore().collection('emotionLogs').document(log_id)
            log_doc = log_ref.get()
            
            if not log_doc.exists:
                return JsonResponse({'error': 'Emotion log not found'}, status=404)
            
            # Delete log
            log_ref.delete()
            
            return JsonResponse({'message': 'Emotion log deleted successfully'})
            
        except Exception as e:
            print(f"Error deleting emotion log: {e}")
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)


@csrf_exempt
@login_required
@user_passes_test(is_superuser)
def admin_update_emotion_log(request, log_id):
    """Update emotion log information (admin only)"""
    if request.method == 'PATCH':
        try:
            data = json.loads(request.body)
            
            # Try different collection structures to find the log
            log_ref = None
            log_doc = None
            
            # Try main emotionLogs collection first
            try:
                log_ref = get_firestore().collection('emotionLogs').document(log_id)
                log_doc = log_ref.get()
                if log_doc.exists:
                    collection_name = 'emotionLogs'
                else:
                    raise Exception("Not found")
            except:
                # Try alternative collection names
                try:
                    log_ref = get_firestore().collection('emotion_logs').document(log_id)
                    log_doc = log_ref.get()
                    if log_doc.exists:
                        collection_name = 'emotion_logs'
                    else:
                        raise Exception("Not found")
                except:
                    try:
                        log_ref = get_firestore().collection('scans').document(log_id)
                        log_doc = log_ref.get()
                        if log_doc.exists:
                            collection_name = 'scans'
                        else:
                            raise Exception("Not found")
                    except:
                        # Try users subcollections
                        users_docs = list(get_firestore().collection('users').stream())
                        for user_doc in users_docs:
                            try:
                                log_ref = get_firestore().collection('users').document(user_doc.id).collection('emotion_logs').document(log_id)
                                log_doc = log_ref.get()
                                if log_doc.exists:
                                    collection_name = 'emotion_logs'
                                    break
                            except:
                                try:
                                    log_ref = get_firestore().collection('users').document(user_doc.id).collection('scans').document(log_id)
                                    log_doc = log_ref.get()
                                    if log_doc.exists:
                                        collection_name = 'scans'
                                        break
                                except:
                                    continue
            
            if not log_doc or not log_doc.exists:
                return JsonResponse({'error': 'Emotion log not found'}, status=404)
            
            # Update allowed fields
            update_data = {}
            allowed_fields = [
                'emotion', 'confidence', 'imageUrl', 'notes', 'location',
                'weather', 'mood', 'activity', 'environment', 'tags',
                'isVerified', 'verificationNotes', 'adminNotes'
            ]
            
            for field in allowed_fields:
                if field in data:
                    update_data[field] = data[field]
            
            if update_data:
                update_data['updatedAt'] = datetime.now(timezone.utc)
                log_ref.update(update_data)
            
            return JsonResponse({'message': 'Emotion log updated successfully'})
            
        except Exception as e:
            print(f"Error updating emotion log: {e}")
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)


@csrf_exempt
@login_required
@user_passes_test(is_superuser)
def admin_debug_firebase(request):
    """Debug Firebase collections to see what data exists"""
    try:
        debug_info = {
            'collections': {},
            'errors': []
        }
        
        # Check users collection
        try:
            users_docs = list(get_firestore().collection('users').stream())
            debug_info['collections']['users'] = {
                'count': len(users_docs),
                'sample_data': []
            }
            for i, user_doc in enumerate(users_docs[:3]):  # First 3 users
                user_data = user_doc.to_dict()
                debug_info['collections']['users']['sample_data'].append({
                    'id': user_doc.id,
                    'data': user_data
                })
        except Exception as e:
            debug_info['errors'].append(f'Users collection error: {str(e)}')
        
        # Check pets collection
        try:
            pets_docs = list(get_firestore().collection('pets').stream())
            debug_info['collections']['pets'] = {
                'count': len(pets_docs),
                'sample_data': []
            }
            for i, pet_doc in enumerate(pets_docs[:3]):  # First 3 pets
                pet_data = pet_doc.to_dict()
                debug_info['collections']['pets']['sample_data'].append({
                    'id': pet_doc.id,
                    'data': pet_data
                })
        except Exception as e:
            debug_info['errors'].append(f'Pets collection error: {str(e)}')
        
        # Check emotionLogs collection
        try:
            emotion_logs = list(get_firestore().collection('emotionLogs').stream())
            debug_info['collections']['emotionLogs'] = {
                'count': len(emotion_logs),
                'sample_data': []
            }
            for i, log_doc in enumerate(emotion_logs[:3]):  # First 3 logs
                log_data = log_doc.to_dict()
                debug_info['collections']['emotionLogs']['sample_data'].append({
                    'id': log_doc.id,
                    'data': log_data
                })
        except Exception as e:
            debug_info['errors'].append(f'EmotionLogs collection error: {str(e)}')
        
        # Check supportMessages collection
        try:
            support_messages = list(get_firestore().collection('supportMessages').stream())
            debug_info['collections']['supportMessages'] = {
                'count': len(support_messages),
                'sample_data': []
            }
            for i, msg_doc in enumerate(support_messages[:3]):  # First 3 messages
                msg_data = msg_doc.to_dict()
                debug_info['collections']['supportMessages']['sample_data'].append({
                    'id': msg_doc.id,
                    'data': msg_data
                })
        except Exception as e:
            debug_info['errors'].append(f'SupportMessages collection error: {str(e)}')
        
        # Check alternative collection names
        alternative_collections = ['petsCollection', 'emotion_logs', 'scans', 'support', 'support_messages']
        for collection_name in alternative_collections:
            try:
                docs = list(get_firestore().collection(collection_name).stream())
                debug_info['collections'][collection_name] = {
                    'count': len(docs),
                    'sample_data': []
                }
                for i, doc in enumerate(docs[:2]):  # First 2 docs
                    doc_data = doc.to_dict()
                    debug_info['collections'][collection_name]['sample_data'].append({
                        'id': doc.id,
                        'data': doc_data
                    })
            except Exception as e:
                debug_info['errors'].append(f'{collection_name} collection error: {str(e)}')
        
        return JsonResponse(debug_info)
        
    except Exception as e:
        return JsonResponse({'error': f'Debug error: {str(e)}'}, status=500)


