## Subscriptions API - Documentation & Postman Guide

This document describes all endpoints under ` /api/subscriptions/ ` and how to test them (e.g. in Postman).

Base URL examples:
- Local dev: `http://127.0.0.1:8000/api/subscriptions/`
- Render:   `https://pet-emotion-detection.onrender.com/api/subscriptions/`

Most endpoints require a **Firebase-authenticated user** (via `Authorization: Bearer <id_token>`), except where noted.

---

### 1. List Available Plans

**Endpoint**
- **GET** `/api/subscriptions/plans`

**Auth**
- No auth required (`AllowAny`).

**Description**
- Returns all **active** subscription plans stored in the `SubscriptionPlan` database table.
- Intended for pricing / subscription selection screens in the client app.

**Sample Response**

```json
{
  "plans": [
    {
      "product_id": "com.petmood.premium.monthly",
      "name": "Premium Monthly",
      "plan_type": "premium",
      "period": "monthly",
      "platform": "apple",
      "price_display": "$4.99 / month"
    },
    {
      "product_id": "com.petmood.premium.annual",
      "name": "Premium Annual",
      "plan_type": "premium",
      "period": "annual",
      "platform": "apple",
      "price_display": "$39.99 / year"
    }
  ]
}
```

**Postman**
- Method: `GET`
- URL: `{{BASE_URL}}/api/subscriptions/plans`
- Headers: none required

---

### 2. Verify Receipt (iOS In-App Purchase)

**Endpoint**
- **POST** `/api/subscriptions/verify-receipt`

**Auth**
- Requires Firebase-authenticated user.
- Header: `Authorization: Bearer <FIREBASE_ID_TOKEN>`

**Description**
- Used by the iOS app after completing a purchase to verify with Apple and save/update the user’s subscription in Firestore.
- Works with Apple App Store Server API using the `transaction_id`.

**Request Body (JSON)**

```json
{
  "product_id": "com.petmood.premium.monthly",
  "transaction_id": "1000001234567890",
  "original_transaction_id": "1000001234567890"
}
```

**Sample Success Response**

```json
{
  "success": true,
  "subscription": {
    "plan_type": "premium",
    "period": "monthly",
    "expires_at": "2026-02-12T17:20:00Z",
    "is_active": true
  }
}
```

**Postman**
- Method: `POST`
- URL: `{{BASE_URL}}/api/subscriptions/verify-receipt`
- Headers:
  - `Content-Type: application/json`
  - `Authorization: Bearer {{FIREBASE_ID_TOKEN}}`
- Body: JSON as above.

---

### 3. Get Current Subscription Status

**Endpoint**
- **GET** `/api/subscriptions/status`

**Auth**
- Requires Firebase-authenticated user.
- Header: `Authorization: Bearer <FIREBASE_ID_TOKEN>`

**Description**
- Returns the user’s **current active subscription** (if any) from Firestore.
- Used by the app on startup or settings screen.

**Sample Response – Active Subscription**

```json
{
  "subscription": {
    "plan_type": "premium",
    "period": "monthly",
    "expires_at": "2026-02-12T17:20:00Z",
    "is_active": true,
    "product_id": "com.petmood.premium.monthly"
  }
}
```

**Sample Response – No Subscription**

```json
{
  "subscription": null
}
```

**Postman**
- Method: `GET`
- URL: `{{BASE_URL}}/api/subscriptions/status`
- Headers:
  - `Authorization: Bearer {{FIREBASE_ID_TOKEN}}`

---

### 4. Restore Purchases

**Endpoint**
- **POST** `/api/subscriptions/restore`

**Auth**
- Requires Firebase-authenticated user.
- Header: `Authorization: Bearer <FIREBASE_ID_TOKEN>`

**Description**
- Returns **all active subscriptions** in Firestore for this user.
- Used when the user taps “Restore Purchases” in the app.

**Request Body**
- Usually just an empty JSON `{}` (no fields required).

**Sample Response**

```json
{
  "subscriptions": [
    {
      "is_active": true,
      "plan_type": "premium",
      "period": "monthly",
      "expires_at": "2026-02-12T17:20:00Z",
      "product_id": "com.petmood.premium.monthly"
    }
  ]
}
```

**Postman**
- Method: `POST`
- URL: `{{BASE_URL}}/api/subscriptions/restore`
- Headers:
  - `Content-Type: application/json`
  - `Authorization: Bearer {{FIREBASE_ID_TOKEN}}`
- Body: `{}` (or empty JSON).

---

### 5. App Store Webhook (Server-to-Server Notifications)

**Endpoint**
- **POST** `/api/subscriptions/webhook`

**Auth**
- `AllowAny` (no Firebase token).
- Should be secured by **Apple’s endpoint IPs** / secret config, not for public use.

**Description**
- Endpoint for **App Store Server Notifications v2**.
- Apple sends a signed JWS (`signedPayload`); backend verifies and, if possible, updates the user subscription in Firestore.

**Request Body (from Apple)**

```json
{
  "signedPayload": "<JWS_TOKEN_FROM_APPLE>"
}
```

**Sample Response**

```json
{
  "received": true,
  "notificationType": "DID_RENEW",
  "subtype": "INITIAL_BUY"
}
```

**Postman (for local testing)**
- Method: `POST`
- URL: `{{BASE_URL}}/api/subscriptions/webhook`
- Headers: `Content-Type: application/json`
- Body: `{ "signedPayload": "..." }` (use test payloads if you have them).

---

### 6. Admin List Subscriptions

**Endpoint**
- **GET** `/api/subscriptions/admin/list?uid=<firebase_uid>`

**Auth**
- Requires Firebase admin privileges (`IsAdminFirebase`).
- Header: `Authorization: Bearer <FIREBASE_ADMIN_ID_TOKEN>`

**Description**
- Admin/debug endpoint to list **all subscriptions** in Firestore for a given user.

**Sample Response**

```json
{
  "uid": "some-firebase-uid",
  "subscriptions": [
    {
      "product_id": "com.petmood.premium.monthly",
      "plan_type": "premium",
      "period": "monthly",
      "expires_at": "2026-02-12T17:20:00Z",
      "is_active": true
    }
  ]
}
```

**Postman**
- Method: `GET`
- URL: `{{BASE_URL}}/api/subscriptions/admin/list?uid={{FIREBASE_UID}}`
- Headers:
  - `Authorization: Bearer {{FIREBASE_ADMIN_ID_TOKEN}}`

---

### Where plans come from

The `/api/subscriptions/plans` endpoint does **not** use Django migrations or a SQL database.

Instead, it derives the list of plans from the hard-coded `PRODUCT_MAPPING` in `subscriptions/views.py`:

```python
PRODUCT_MAPPING = {
    "com.petmood.premium.monthly": ("premium", "monthly"),
    "com.petmood.premium.annual": ("premium", "annual"),
    "com.petmood.family.monthly": ("family", "monthly"),
    "com.petmood.family.annual": ("family", "annual"),
}
```

For each `product_id`, it returns a plan object like:

- `product_id`: the store product id (used by Apple)
- `name`: e.g. `Premium Monthly` (generated from type + period)
- `plan_type`: `premium` or `family`
- `period`: `monthly` or `annual`
- `platform`: currently `"apple"`
- `price_display`: empty string by default (can be set in the client app)

So you **do not need** any `makemigrations` / `migrate` for plans; they are defined in code and sent directly via the API.

