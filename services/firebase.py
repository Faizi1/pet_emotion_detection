from typing import Optional
import firebase_admin
from firebase_admin import credentials, auth, firestore, storage
from django.conf import settings
import os
from google.cloud.storage import Bucket
import json

_initialized: bool = False


def initialize_firebase_if_needed() -> None:
    global _initialized
    if _initialized:
        return

    cred: Optional[credentials.Base] = None

    # Try to get JSON credentials from environment variable
    firebase_json = os.getenv("FIREBASE_CREDENTIALS_JSON")

    if settings.FIREBASE_CREDENTIALS_PATH and os.path.exists(settings.FIREBASE_CREDENTIALS_PATH):
        # Local development (file-based)
        cred = credentials.Certificate(settings.FIREBASE_CREDENTIALS_PATH)
    elif firebase_json:
        # Production on Render (env-based)
        firebase_dict = json.loads(firebase_json)
        cred = credentials.Certificate(firebase_dict)
    else:
        # Fallback to default credentials if neither provided
        cred = credentials.ApplicationDefault()

    firebase_admin.initialize_app(cred, {
        'projectId': getattr(settings, 'FIREBASE_PROJECT_ID', None),
        'storageBucket': getattr(settings, 'FIREBASE_STORAGE_BUCKET', None),
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
    if not (settings.FIREBASE_STORAGE_BUCKET or '').strip():
        raise RuntimeError('Firebase Storage is not configured.')
    return storage.bucket()
