import uuid
import os
from typing import Optional
from django.conf import settings
from .firebase import get_bucket


def upload_bytes(content: bytes, content_type: str, path_prefix: str = 'uploads/') -> str:
    # Try Firebase first
    try:
        bucket = get_bucket()
        try:
            filename = f"{path_prefix}{uuid.uuid4().hex}"
            blob = bucket.blob(filename)
            blob.upload_from_string(content, content_type=content_type)
            blob.make_public()
            return blob.public_url
        except Exception as e:
            print(f"Upload error (Firebase): {str(e)}")
    except Exception as e:
        print(f"Firebase Storage unavailable: {str(e)}")

    # Fallback to local disk
    try:
        # Ensure subdir exists under MEDIA_ROOT
        safe_prefix = path_prefix.strip('/\\')
        subdir_path = os.path.join(settings.MEDIA_ROOT, safe_prefix)
        os.makedirs(subdir_path, exist_ok=True)

        # Determine extension by content type (basic)
        ext = ''
        if content_type.startswith('image/'):
            ext = '.' + content_type.split('/')[-1]
        elif content_type.startswith('audio/'):
            ext = '.' + content_type.split('/')[-1]

        filename_only = f"{uuid.uuid4().hex}{ext}"
        rel_path = os.path.join(safe_prefix, filename_only).replace('\\', '/')
        abs_path = os.path.join(settings.MEDIA_ROOT, rel_path)

        with open(abs_path, 'wb') as f:
            f.write(content)

        # Public URL served by Django in DEBUG via MEDIA_URL
        url = settings.MEDIA_URL.rstrip('/') + '/' + rel_path
        return url
    except Exception as e:
        print(f"Upload error (local): {str(e)}")
        return ''


