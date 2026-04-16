# Subscriptions Complete Implementation Guide (Backend + Frontend)

This is the practical source of truth for how subscriptions currently work in this backend and what frontend must do.

Reviewed against:
- `subscriptions/urls.py`
- `subscriptions/views.py`
- `subscriptions/firestore.py`
- `HARIS-IAP_VERIFY_RECEIPT_AND_BACKEND.md`
- `HARIS-BACKEND_IAP_REQUIREMENTS (1).md`

---

## 1) Current subscription architecture

### Route base

Project routes include `services.urls` under `/api/`, and `services.urls` includes `subscriptions.urls` under `subscriptions/`.

So subscription endpoints are:

- `POST /api/subscriptions/verify-receipt`
- `GET /api/subscriptions/status`
- `POST /api/subscriptions/restore`
- `POST /api/subscriptions/webhook`
- `GET /api/subscriptions/admin/list`
- `GET /api/subscriptions/plans`

### Data storage

Subscriptions are stored in Firestore under:

- `users/{uid}/subscriptions/{transaction_id}`

Saved fields include:
- `uid`
- `product_id`
- `plan_type`
- `period`
- `transaction_id`
- `original_transaction_id`
- `expires_at` (ISO UTC string)
- `is_active`
- `created_at`, `updated_at`

---

## 2) Endpoint-by-endpoint behavior (what actually happens)

## `GET /api/subscriptions/plans`
- Public endpoint (`AllowAny`)
- Returns plans from hardcoded `PRODUCT_MAPPING`
- `price_display` is currently empty string; frontend should show App Store localized price from StoreKit

## `POST /api/subscriptions/verify-receipt`
- Requires Firebase authenticated user
- Validates request body with serializer:
  - required: `product_id`, `transaction_id`
  - optional: `original_transaction_id`, `receipt_data`
- Calls Apple App Store Server API by `transaction_id`
- Verifies Apple signed transaction info and `bundleId`
- Ensures returned `productId` matches request product
- Saves subscription in Firestore
- Returns:
  - `success: true`
  - normalized subscription (`plan_type`, `period`, `expires_at`, `is_active`)

## `GET /api/subscriptions/status`
- Requires Firebase auth
- Reads active subscription for current uid
- If none active: `{"subscription": null}`
- If active: subscription object
- Includes fallback logic if Firestore index is still building (no more hard crash from that index issue)

## `POST /api/subscriptions/restore`
- Requires Firebase auth
- Returns all active subscriptions for current user from Firestore

## `POST /api/subscriptions/webhook`
- `AllowAny` (Apple server callback)
- Expects `signedPayload` (App Store Server Notifications v2)
- Decodes and verifies JWS
- Attempts to parse `signedTransactionInfo`
- If `appAccountToken` is present (and convention is `appAccountToken == Firebase uid`), updates that user’s subscription
- Always returns 200 when payload is handled (important so Apple stops retries)

## `GET /api/subscriptions/admin/list?uid=<firebase_uid>`
- Admin only (`IsAdminFirebase`)
- Lists all subscription docs for that user

---

## 3) Product IDs currently configured

- `com.petmood.premium.monthly` -> `premium`, `monthly`
- `com.petmood.premium.annual` -> `premium`, `annual`
- `com.petmood.family.monthly` -> `family`, `monthly`
- `com.petmood.family.annual` -> `family`, `annual`

These must exactly match App Store Connect IAP product IDs.

---

## 4) HARIS docs vs actual backend (important)

The HARIS docs describe `signed_transaction_jws` support in request payload.

Current backend reality:
- Serializer does **not** currently include `signed_transaction_jws`
- Backend verifies by `transaction_id` with Apple Server API (this is fine and secure)

So:
- Frontend should send at minimum `product_id` + `transaction_id` (works now)
- If you want strict contract parity with HARIS docs, backend can be extended to accept `signed_transaction_jws` too

---

## 5) Apple compliance status for subscription backend

Backend verification flow is aligned with Apple best practice:
- Server-side verification with App Store Server API
- Webhook endpoint exists for server notifications
- Subscription status/restore endpoints exist

But Apple 3.1.2(c) rejection is mostly about **app UI + metadata** (not just backend):

Must be visible in app subscription screen:
- Subscription title
- Subscription length
- Subscription price
- Privacy Policy link
- Terms of Use (EULA) link

Must be present in App Store Connect:
- Privacy Policy field link
- EULA link (App Description or EULA field)

---

## 6) Frontend integration flow (recommended)

1. Frontend loads products from StoreKit (`react-native-iap`) for localized price.
2. User buys subscription.
3. Frontend calls:
   - `POST /api/subscriptions/verify-receipt`
   - body includes at least:
     - `product_id`
     - `transaction_id`
     - optional `original_transaction_id`
4. On success:
   - unlock premium features
   - call `GET /api/subscriptions/status` on app start/profile
5. Restore button:
   - `POST /api/subscriptions/restore`
6. Backend webhook keeps status in sync for renewals/cancellations.

---

## 7) Postman quick test checklist

1. `GET /api/subscriptions/plans`
2. `POST /api/subscriptions/verify-receipt` with real sandbox transaction id
3. `GET /api/subscriptions/status` (expect active subscription)
4. `POST /api/subscriptions/restore` (expect list)
5. `POST /api/subscriptions/webhook` with test signed payload (if available)
6. `GET /api/subscriptions/admin/list?uid=<uid>` with admin token

---

## 8) What to tell frontend developer (short version)

Use this exact contract now:
- Required verify payload: `product_id`, `transaction_id`
- Optional: `original_transaction_id`
- Status endpoint returns `subscription` or `null`
- Restore endpoint returns array of active subscriptions
- UI must show title/duration/price + privacy policy + terms links for Apple

If frontend already sends `signed_transaction_jws`, backend currently ignores it; this does not break flow as long as `transaction_id` is sent.

