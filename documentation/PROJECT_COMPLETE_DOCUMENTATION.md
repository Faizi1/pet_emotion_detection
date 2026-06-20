# PetMood Backend - Complete Project Documentation

This document is the current source of truth for how the backend works right now.

---

## 1) Project Overview

`pet_emotion_detection` is a Django + DRF backend for the PetMood mobile application.

It provides:
- user authentication and account management (Firebase-based)
- pet profile management
- emotion scans for pets (image/audio)
- scan history and analytics
- community feed (posts, comments, likes, shares, reports, block)
- support/helpdesk system (user + admin panel)
- iOS subscription verification and status APIs
- custom Firestore-backed admin panel

Primary stack:
- Django / Django REST Framework
- Firebase Admin SDK (Auth + Firestore + Storage)
- Nyckel (image emotion classification)
- AssemblyAI (audio sentiment/emotion signals)
- Resend (transactional email)
- Multi-provider SMS service support (Telnyx)

---

## 2) Repository Structure

### `pet_emotion_detection/`
- Django project config and root routing.
- Key files:
  - `pet_emotion_detection/settings.py`
  - `pet_emotion_detection/urls.py`
  - `pet_emotion_detection/wsgi.py`
  - `pet_emotion_detection/asgi.py`

### `services/`
Main app containing most product APIs and integrations.

Key files:
- `services/views.py` - core APIs (auth, pets, scans, community, support, admin APIs)
- `services/urls.py` - API routes included under `/api/`
- `services/serializers.py` - request/response validation
- `services/auth.py` - Firebase token authentication classes
- `services/permissions.py` - admin permission checks
- `services/firebase.py` - Firebase app/auth/firestore bootstrap
- `services/storage.py` - media upload logic
- `services/sms_service.py` - SMS providers and fallback logic
- `services/email_service.py` - email helpers (Resend)
- `services/admin.py` - custom Django admin-panel connected to Firestore
- `services/nyckel_service.py` - Nyckel image classifier integration
- `services/assemblyai_service.py` - AssemblyAI audio sentiment integration

### `subscriptions/`
iOS subscription (Apple) implementation.

Key files:
- `subscriptions/urls.py`
- `subscriptions/views.py`
- `subscriptions/firestore.py`
- `subscriptions/apple.py`
- `subscriptions/serializers.py`

### `templates/admin/`
Custom admin UI templates for `/admin-panel/`.

Key templates:
- `templates/admin/base_site.html`
- `templates/admin/firestore_dashboard.html`
- `templates/admin/firestore_users.html`
- `templates/admin/firestore_pets.html`
- `templates/admin/firestore_logs.html`
- `templates/admin/firestore_community.html`
- `templates/admin/firestore_support.html`
- `templates/admin/firestore_user_edit.html`
- `templates/admin/firestore_pet_edit.html`

### `documentation/`
Project docs, implementation guides, and operational notes.

### `postman/`
Postman collections for API testing.

---

## 3) Routing and Entry Points

Root router: `pet_emotion_detection/urls.py`

Main mounts:
- `/admin/` -> default Django admin
- `/admin-panel/` -> custom Firestore dashboard/admin site
- `/api/` -> `services.urls`
- `/swagger/`, `/redoc/` -> API documentation UI

Important:
- `subscriptions.urls` is included from `services/urls.py` under:
  - `/api/subscriptions/...`

---

## 4) Authentication and Authorization Model

### API authentication
- DRF default auth class: `services.auth.FirebaseAuthentication`
- Expects: `Authorization: Bearer <firebase_id_token>`
- Token verified using Firebase Admin SDK.

### API permissions
- Default: authenticated user required.
- Some endpoints override to `AllowAny` (example: auth bootstrapping, support send, certain utility endpoints, plans/webhook endpoints for subscriptions).

### Admin APIs
- Permission class `IsAdminFirebase` for admin-only API actions (claim based).

### Custom admin panel
- `/admin-panel/` uses Django staff session auth (`staff_member_required`), separate from Firebase Bearer auth.

---

## 5) Core Functional Modules

## 5.1 Auth and Account

Implemented in `services/views.py` and routed from `services/urls.py`.

Includes:
- token verify
- email/password login guidance flow
- registration with phone OTP verification
- OTP resend / verify
- forgot/reset password OTP flows
- profile update
- account deletion
- social sign in:
  - Google sign-in
  - Apple sign-in

### Temporary registration flow (`temp_registrations`)
- Collection stores pending registration state before OTP verification.
- Fields typically include:
  - `name`, `email`, `number`, `password`
  - `otp`, `otpCreatedAt`, `attempts`
- OTP validity: 10 minutes (checked during verify/resend calls).
- Not auto-cleaned by timer in current code; cleanup happens on follow-up API actions.

---

## 5.2 Pets

Endpoints support:
- create/list pets
- get/update/delete pet
- optional pet image uploads

Stored under Firestore subcollection:
- `users/{uid}/pets/{petId}`

---

## 5.3 Scan Pipeline (Pet Emotion Detection)

Scan creation endpoint is implemented in `services/views.py`.

Current scan provider behavior:
- **Image scans:** Nyckel integration (`services/nyckel_service.py`)
- **Audio scans:** AssemblyAI integration (`services/assemblyai_service.py`)

### AI consent gate
Before third-party AI processing, backend checks user consent payload (`aiConsent`) and can reject with disclosure info if consent missing.

### Scan output
Result is written to:
- `users/{uid}/emotion_logs/{logId}`

Typical fields:
- `emotion`
- `confidence`
- `topEmotions`
- `mediaUrl`
- `mediaType`
- `analysisMethod`
- `aiDetectorType`
- `timestamp`
- `petId`

### Trial scan limit
If user is on active subscription trial, backend enforces daily scan cap (currently 7/day UTC).

---

## 5.4 History and Analytics

History endpoints provide:
- list emotion logs with filters/search
- detail view by log id
- delete log

Admin analytics endpoints exist for aggregate counters and charts (users/pets/scans/support/moderation metrics).

---

## 5.5 Community and Moderation

Community features include:
- posts list/create/detail/delete
- my posts
- post like/share
- comments create/list/delete
- report post/comment
- block user

Moderation:
- report listing
- moderation action endpoint with options like dismiss/review/remove/suspend.

Data is stored across:
- `community_posts`
- `community_posts/{postId}/comments`
- `community_posts/{postId}/likes`
- `community_posts/{postId}/shares`
- `content_reports`
- `users/{uid}/blocks`

---

## 5.6 Support / Helpdesk

### User-side APIs
- `POST /api/support/send`
- `GET /api/support/my`

When user submits support:
1. backend writes `support_messages/{messageId}` in Firestore
2. returns success and `supportId`
3. attempts admin/team email notification via Resend

### Admin panel support workflow (`/admin-panel/support/`)
Implemented by `services/admin.py` + `templates/admin/firestore_support.html`.

Admin can:
- view full details (modal)
- reply to message
- update status
- delete message

Action routes:
- `POST /admin-panel/support/<message_id>/reply/`
- `POST /admin-panel/support/<message_id>/update-status/`
- `POST /admin-panel/support/<message_id>/delete/`

Reply action can also send email to user via Resend.

Support status model currently accepts:
- modern values: `new`, `read`, `replied`
- legacy-compatible values: `pending`, `in_progress`, `resolved`, `closed`

---

## 5.7 Subscriptions (Apple iOS IAP)

Routes under:
- `/api/subscriptions/...`

Main endpoints:
- `POST /verify-receipt`
- `GET /status`
- `POST /restore`
- `POST /cancel`
- `POST /webhook`
- `GET /admin/list`
- `GET /plans`

### Verification flow
`verify-receipt` validates product + transaction with Apple data (JWS and/or transaction API), then saves subscription document.

### Webhook flow
`webhook` receives App Store Server Notification payloads, decodes signed payload(s), and updates stored subscription when user mapping is possible.

### Status semantics (current)
`/api/subscriptions/status` returns `status` inside `subscription`:
- `trialing` - active and trial days left
- `active` - active non-trial (or trial completed but still valid)
- `canceled` - active entitlement but auto-renew turned off
- `expired` - previous subscription exists but not active
- `subscription: null` when no subscription exists

### Cancel endpoint
`/cancel` is informational for Apple flow; backend does not directly cancel Apple subscriptions. User cancels from Apple subscription settings.

---

## 6) Data Model (Firestore)

Primary collections/subcollections in current code:

- `users/{uid}`
  - profile and settings (name/email/number/phoneVerified/location/photo, consent, moderation flags)

- `users/{uid}/pets/{petId}`
  - pet profile data

- `users/{uid}/emotion_logs/{logId}`
  - scan history records

- `users/{uid}/subscriptions/{transactionId}`
  - subscription state: product, plan, period, expiry, active, trial/offer/env, auto-renew metadata

- `users/{uid}/blocks/{targetUid}`
  - block relations

- `community_posts/{postId}`
  - post data, with subcollections:
    - `comments`
    - `likes`
    - `shares`

- `content_reports/{reportId}`
  - moderation report records

- `support_messages/{messageId}`
  - support ticket records

- `temp_registrations/{phone}`
  - pending registration + OTP

- `temp_password_resets/{phone}`
  - pending password reset OTP state

---

## 7) Admin Panel (`/admin-panel/`)

Custom admin class: `FirestoreDashboardAdmin` in `services/admin.py`.

Sections:
- Dashboard
- Users
- Pets
- Logs
- Community
- Support

Features:
- Firestore-backed tables and filters
- edit/delete actions for user/pet/log/support
- support modals and workflows
- statistics cards and charts on dashboard

---

## 8) Environment Configuration

Main environment variables grouped by concern:

### Core Django
- `DJANGO_SECRET_KEY`
- `DEBUG`

### Firebase
- `FIREBASE_PROJECT_ID`
- `FIREBASE_STORAGE_BUCKET`
- `FIREBASE_CREDENTIALS_PATH`
- `FIREBASE_CREDENTIALS_JSON`
- `FIREBASE_WEB_API_KEY`

### Apple IAP
- `APPLE_IAP_KEY_ID`
- `APPLE_IAP_ISSUER_ID`
- `APPLE_IAP_BUNDLE_ID`
- `APPLE_IAP_PRIVATE_KEY`
- `APPLE_IAP_PRIVATE_KEY_PATH`
- `APPLE_IAP_USE_SANDBOX`

### AI providers
- `NYCKEL_BEARER_TOKEN`
- `NYCKEL_CLIENT_ID`
- `NYCKEL_CLIENT_SECRET`
- `ASSEMBLYAI_API_KEY`
- `LIGHTWEIGHT_MODE`

### Email / Support
- `RESEND_API_KEY`
- `EMAIL_FROM`
- `SUPPORT_NOTIFICATION_EMAILS`
- `BLOCK_USER_EMAIL_ENABLED`

### SMS providers
- `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_PHONE_NUMBER`
- `TELGORITHM_API_KEY`, `TELGORITHM_SENDER_ID`, `TELGORITHM_API_URL`, `USE_TELGORITHM`
- `VONAGE_API_KEY`, `VONAGE_API_SECRET`, `VONAGE_SENDER_ID`, `USE_VONAGE`
- `TELNYX_API_KEY`, `TELNYX_MESSAGING_PROFILE_ID`, `TELNYX_SENDER_ID`, `USE_TELNYX`, `USE_ENHANCED_HYBRID`

---

## 9) Operational Notes and Known Caveats

- Some older docs/Postman examples may not exactly match current route paths/response shapes. Code routes are authoritative.
- `temp_registrations` is not hard-TTL auto-delete by default; expiry is enforced on verify/resend requests.
- Support status values include both current and legacy values (intentional compatibility).
- Certain analytics/admin queries use broad Firestore scans; may need optimization at larger scale.
- Media storage has fallback behavior when Firebase storage is unavailable.

---

## 10) Current “How PetMood Works” Summary

1. User signs up/signs in (Firebase auth token used for API access).
2. User adds pet profile(s).
3. User sends image/audio scan request.
4. Backend validates quota/consent/media and calls AI provider:
   - Nyckel for image
   - AssemblyAI for audio
5. Backend stores result in `emotion_logs`.
6. User sees history/analytics from stored logs.
7. User can use community features (post/comment/like/share/report/block).
8. User can contact support from app; support appears in admin panel and is managed there.
9. Subscription APIs manage Apple entitlement state and return lifecycle status to app.

---

## 11) Suggested Future Improvements (Optional)

- Add Firestore TTL (`expiresAt`) for automatic cleanup of temp OTP docs.
- Standardize support status vocabulary to one set only.
- Align all docs/Postman collections to current live routes.
- Add query/index optimizations for analytics-heavy endpoints.
- Harden temporary password handling in OTP registration flow.

