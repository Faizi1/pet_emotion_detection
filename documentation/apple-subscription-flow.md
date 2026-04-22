# Apple Subscriptions: Frontend Integration Guide

This document explains exactly how frontend should use

## Important platform rules (Apple)

- Apple decides trial eligibility (based on App Store account + subscription group history).
- Apple controls real subscription lifecycle events (renew, expire, billing fail, refund, etc.).
- Backend cannot directly cancel Apple subscriptions for users.

## Backend APIs used by frontend

- `POST /api/subscriptions/verify-receipt`
- `GET /api/subscriptions/status`
- `POST /api/subscriptions/restore`
- `POST /api/subscriptions/cancel`
- `GET /api/subscriptions/plans`
- `POST /api/scans` (trial daily limit is enforced here)

## 1) Trial flow (frontend)

### Step A: Show plans
Call `GET /api/subscriptions/plans` to render available plans and trial label.

Notes:
- `trial_offer` is for UI guidance.
- Eligibility is finalized by Apple at purchase time.

### Step B: User buys in iOS app
Use StoreKit purchase flow in app. After success, send receipt data to backend:

`POST /api/subscriptions/verify-receipt`

Typical body:
- `product_id`
- `transaction_id`
- `original_transaction_id`
- optional `signed_transaction_jws` (recommended)

### Step C: Read canonical entitlement
Call `GET /api/subscriptions/status` and use that response as source of truth.

Fields:
- `is_active`
- `is_trial`
- `trial_days`
- `trial_days_left`
- `expires_at`

## 2) How trial countdown works

- `trial_days_left` is computed by backend from `expires_at`.
- It updates whenever frontend calls `status` (or `restore`).
- If user has 2 days left, response returns `trial_days_left: 2`.

Recommended frontend behavior:
- Call `status` on app launch, after purchase, and when subscription screen opens.
- Also refresh on app resume.

## 3) Trial scan limit (7/day)

Backend enforces this on `POST /api/scans`:

- Applies only when active subscription is trial (`is_trial=true` and not expired).
- Limit is `7` scans per UTC day.
- On 8th scan the same day, backend returns `429`.

Example limit response:

```json
{
  "detail": "Daily trial scan limit reached",
  "limit": 7,
  "used": 7,
  "remaining": 0,
  "resetsAt": "2026-04-21T00:00:00Z"
}
```

Frontend should show user-friendly message and the reset timing.

## 4) After trial expires

- Entitlement becomes inactive.
- `GET /api/subscriptions/status` returns no active access and expired context.
- Premium features should be locked until user has an active subscription again.

If Apple renews into paid period, subsequent verification/webhook update should make access active again.

## 5) Cancel API usage (frontend)

Endpoint: `POST /api/subscriptions/cancel`

What it does:
- Returns instructions + Apple manage subscription URL.
- Does not cancel directly (Apple policy).

Frontend action:
- Call this API when user taps "Cancel Subscription".
- Open `manageSubscriptionUrl` in browser/webview.
- Explain user must complete cancellation in Apple subscription settings.

Example response:

```json
{
  "success": true,
  "platform": "apple",
  "canCancelDirectlyFromBackend": false,
  "message": "Apple requires user-initiated cancellation from App Store subscription settings.",
  "manageSubscriptionUrl": "https://apps.apple.com/account/subscriptions",
  "subscription": {}
}
```

## 6) Recommended frontend call sequence

1. Open paywall -> `GET /plans`
2. Complete StoreKit purchase in app
3. Send purchase to backend -> `POST /verify-receipt`
4. Fetch entitlement -> `GET /status`
5. On restore action -> `POST /restore`, then `GET /status`
6. On cancel action -> `POST /cancel`, open returned URL
7. On app resume / subscription screen open -> `GET /status`

## 7) Production safety notes

- Do not rely only on client-side trial checks.
- Backend status is the entitlement source of truth.
- Avoid exposing full decoded webhook/JWS payloads in production responses.
