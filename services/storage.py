import uuid
from typing import Optional
from .firebase import get_bucket


def upload_bytes(content: bytes, content_type: str, path_prefix: str = 'uploads/') -> str:
    try:
        bucket = get_bucket()
    except Exception:
        # Storage disabled; return empty string so callers can proceed without media
        return ''
    filename = f"{path_prefix}{uuid.uuid4().hex}"
    blob = bucket.blob(filename)
    blob.upload_from_string(content, content_type=content_type)
    blob.make_public()
    return blob.public_url


