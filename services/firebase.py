from typing import Optional
import firebase_admin
from firebase_admin import credentials, auth, firestore, storage
from django.conf import settings
import os
from google.cloud.storage import Bucket

_initialized: bool = False


def initialize_firebase_if_needed() -> None:
    global _initialized
    if _initialized:
        return
    cred: Optional[credentials.Base] = None
    if settings.FIREBASE_CREDENTIALS_PATH and os.path.exists(settings.FIREBASE_CREDENTIALS_PATH):
        cred = credentials.Certificate(settings.FIREBASE_CREDENTIALS_PATH)
    else:
        cred = credentials.ApplicationDefault()
    firebase_admin.initialize_app(cred, {
        'projectId': settings.FIREBASE_PROJECT_ID or None,
        'storageBucket': settings.FIREBASE_STORAGE_BUCKET or None,
    })
    _initialized = True


def get_auth() -> auth:
    initialize_firebase_if_needed()
    return auth


def get_firestore() -> firestore.Client:
    initialize_firebase_if_needed()
    return firestore.client()


def get_bucket() -> Bucket:
    initialize_firebase_if_needed()
    # If storage bucket not configured, raise to allow callers to handle gracefully
    if not (settings.FIREBASE_STORAGE_BUCKET or '').strip():
        raise RuntimeError('Firebase Storage is not configured.')
    return storage.bucket()


