from datetime import datetime, timezone, date
import random
from typing import Any, Dict, List
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from google.api_core.exceptions import FailedPrecondition
from .auth import FirebaseUser
from .firebase import get_auth, get_firestore
from .serializers import (
    UserSerializer,
    PetSerializer,
    EmotionScanRequestSerializer,
    EmotionScanResponseSerializer,
    EmotionLogSerializer,
    RegisterSerializer,
    SendOtpSerializer,
    VerifyOtpSerializer,
    LoginSerializer,
    ForgotPasswordSerializer,
)
from .storage import upload_bytes
from .permissions import IsAdminFirebase


def _pets_collection(uid: str):
    return get_firestore().collection('users').document(uid).collection('pets')


def _logs_collection(uid: str):
    return get_firestore().collection('users').document(uid).collection('emotion_logs')


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

@api_view(['POST'])
@permission_classes([AllowAny])
def register(request):
    ser = RegisterSerializer(data=request.data)
    ser.is_valid(raise_exception=True)
    data = ser.validated_data
    # Create Firebase Auth user with email/password
    try:
        user_record = get_auth().create_user(
            email=data['email'], password=data['password'], display_name=data['name']
        )
    except Exception as e:
        return Response({'detail': str(e)}, status=status.HTTP_400_BAD_REQUEST)
    # Store phone number and profile in Firestore
    db = get_firestore()
    db.collection('users').document(user_record.uid).set({
        'name': data['name'],
        'email': data['email'],
        'number': data['number'],
        'createdAt': datetime.now(timezone.utc),
        'phoneVerified': False,
    }, merge=True)
    return Response({'uid': user_record.uid}, status=status.HTTP_201_CREATED)


@api_view(['POST'])
@permission_classes([AllowAny])
def send_otp(request):
    # For MVP: generate OTP and store in Firestore; in production integrate SMS provider (Twilio, etc.)
    ser = SendOtpSerializer(data=request.data)
    ser.is_valid(raise_exception=True)
    uid = ser.validated_data['uid']
    code = f"{random.randint(100000, 999999)}"
    db = get_firestore()
    db.collection('users').document(uid).set({
        'otp': code,
        'otpCreatedAt': datetime.now(timezone.utc),
    }, merge=True)
    # TODO: integrate SMS sending to stored 'number'
    return Response({'sent': True})


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


@api_view(['POST'])
@permission_classes([AllowAny])
def login_email_password(request):
    # Firebase Admin SDK cannot mint ID tokens from email/password; clients should sign in via Firebase Client SDK.
    # This endpoint exists only to validate payload and guide clients.
    ser = LoginSerializer(data=request.data)
    ser.is_valid(raise_exception=True)
    return Response({'detail': 'Use Firebase client SDK to sign in and send ID token in Authorization header.'})


@api_view(['POST'])
@permission_classes([AllowAny])
def forgot_password(request):
    ser = ForgotPasswordSerializer(data=request.data)
    ser.is_valid(raise_exception=True)
    email = ser.validated_data['email']
    try:
        link = get_auth().generate_password_reset_link(email)
    except Exception as e:
        return Response({'detail': str(e)}, status=status.HTTP_400_BAD_REQUEST)
    # Send email via your provider, or return link for MVP
    return Response({'resetLink': link})


@api_view(['GET'])
def me(request):
    user: FirebaseUser = request.user  # type: ignore
    return Response(UserSerializer(user.__dict__).data)


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


@api_view(['POST'])
def scans_create(request):
    user: FirebaseUser = request.user  # type: ignore
    ser = EmotionScanRequestSerializer(data=request.data)
    ser.is_valid(raise_exception=True)
    file = ser.validated_data['file']
    media_type = ser.validated_data['mediaType']
    pet_id = ser.validated_data['petId']
    content_type = getattr(file, 'content_type', 'application/octet-stream')
    media_url = upload_bytes(file.read(), content_type, path_prefix=f"{user.uid}/{media_type}/")

    # AI model stub: randomly select an emotion
    emotion = random.choice(['happy', 'sad', 'anxious', 'excited', 'neutral'])
    confidence = round(random.uniform(0.6, 0.99), 2)

    log = {
        'petId': pet_id,
        'timestamp': datetime.now(timezone.utc),
        'emotion': emotion,
        'confidence': confidence,
        'mediaUrl': media_url,
    }
    _logs_collection(user.uid).add(log)

    resp = {
        'emotion': emotion,
        'confidence': confidence,
        'mediaUrl': media_url,
        'petId': pet_id,
    }
    return Response(EmotionScanResponseSerializer(resp).data, status=status.HTTP_201_CREATED)


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
    for d in docs_list:
        item = d.to_dict()
        item['id'] = d.id
        logs.append(item)
    return Response(logs)


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


