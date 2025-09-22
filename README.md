# Pet Emotion Detection Backend (Django + Firebase)

Backend API for a React Native app to analyze pet emotions. Uses Django REST Framework with Firebase (Auth, Firestore, Storage).

## Setup

1) Create and activate venv (Windows):
```
python -m venv venv
venv\\Scripts\\activate
```

2) Install requirements:
```
pip install -r requirements.txt
```

3) Environment variables (create `.env` or set in your host):
```
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_STORAGE_BUCKET=your-project-id.appspot.com
FIREBASE_CREDENTIALS_PATH=absolute\\path\\to\\service_account.json
DJANGO_SECRET_KEY=your-django-secret
DEBUG=1
ALLOWED_HOSTS=*
```

4) Run migrations and start server:
```
python manage.py migrate
python manage.py runserver 0.0.0.0:8000
```

## API

- `POST /api/auth/verify` - Exchange Firebase ID token for session (stateless verify)
- `GET /api/me` - Current user profile
- `GET/POST /api/pets` - List/Create pets
- `GET/PUT/PATCH/DELETE /api/pets/{id}` - Pet CRUD
- `POST /api/scans` - Upload audio/image for analysis (AI stub)
- `GET /api/history?petId=...` - Emotion logs
- `GET /api/admin/analytics` - Admin analytics (requires admin claim)

## Firebase

Provide a service account key with Firestore + Storage permissions. Auth is verified via Firebase ID tokens from the mobile app.
