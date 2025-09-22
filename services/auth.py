from typing import Optional, Tuple
from rest_framework.authentication import BaseAuthentication
from rest_framework import exceptions
from django.conf import settings
from .firebase import get_auth


class FirebaseUser:
    def __init__(self, uid: str, email: Optional[str], name: Optional[str], is_admin: bool):
        self.uid = uid
        self.email = email
        self.name = name
        self.is_admin = is_admin

    @property
    def is_authenticated(self) -> bool:
        return True


class FirebaseAuthentication(BaseAuthentication):
    keyword = 'Bearer'

    def authenticate(self, request) -> Optional[Tuple[FirebaseUser, None]]:
        auth_header = request.META.get('HTTP_AUTHORIZATION', '')
        if not auth_header:
            return None
        parts = auth_header.split()
        if len(parts) != 2 or parts[0] != self.keyword:
            raise exceptions.AuthenticationFailed('Invalid Authorization header')
        id_token = parts[1]
        try:
            decoded = get_auth().verify_id_token(id_token)
        except Exception:
            raise exceptions.AuthenticationFailed('Invalid Firebase token')
        uid = decoded.get('uid') or decoded.get('sub')
        email = decoded.get('email')
        name = decoded.get('name') or decoded.get('firebase', {}).get('sign_in_provider')
        is_admin = bool(decoded.get('admin') or decoded.get('claims', {}).get('admin'))
        return FirebaseUser(uid, email, name, is_admin), None


