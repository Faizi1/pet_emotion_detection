# Block user — email notification (Resend, free tier)

When a user blocks another user, the backend can email the **blocked** user a neutral notice.  
Email is sent via **Resend** ([resend.com](https://resend.com)) — free tier is enough for testing and low traffic (see their current limits on the pricing page).

## Environment variables

Add to `.env` or your host (e.g. Render):

| Variable | Required | Description |
|----------|----------|-------------|
| `RESEND_API_KEY` | Yes, to send email | API key from Resend dashboard (`re_...`) |
| `EMAIL_FROM` | Recommended | Sender, e.g. `PetMood <onboarding@resend.dev>` for quick tests, or `PetMood <noreply@yourdomain.com>` after you verify a domain in Resend |
| `BLOCK_USER_EMAIL_ENABLED` | No | Default `1`. Set `0` to disable emails (blocking still works) |

If `RESEND_API_KEY` is missing, the block API **still returns success**; no email is sent (check server logs).

## Resend setup (quick)

1. Create a free account at [resend.com](https://resend.com).
2. **API Keys** → create a key → copy `RESEND_API_KEY`.
3. For first tests, Resend allows sending **from** `onboarding@resend.dev` to **your own verified inbox** (follow Resend’s onboarding — they may require verifying your recipient email).
4. For production, add and verify your domain in Resend, then set e.g. `EMAIL_FROM=PetMood <noreply@yourdomain.com>`.

## API response

`POST /api/services/community/users/<target_uid>/block` now includes:

```json
{
  "blocked": true,
  "targetUid": "...",
  "reportId": "...",
  "blockNotificationEmailSent": true
}
```

- `blockNotificationEmailSent`: `true` if Resend accepted the send; `false` if no API key, email disabled, blocked user has no Firebase email, or Resend returned an error.

## How to test the full flow

### 1) Two test accounts

- **User A** (blocker): sign in, get Firebase ID token.
- **User B** (blocked): must have an **email** on the Firebase Auth account (so `get_user(B)` returns an email).

### 2) Configure backend

Set `RESEND_API_KEY` and optionally `EMAIL_FROM` on the server, restart.

### 3) Call block API

```http
POST /api/services/community/users/<USER_B_UID>/block
Authorization: Bearer <USER_A_TOKEN>
Content-Type: application/json

{ "reason": "Test block" }
```

### 4) Verify

- Response: `blocked: true`, and ideally `blockNotificationEmailSent: true`.
- **User B’s inbox**: neutral email from PetMood (may land in spam first time).
- **Firestore**: `users/<A>/blocks/<B>` exists; `content_reports` has a `block_user` entry.

### 5) Feed behavior (unchanged)

- As **User A**, refresh community feed — **User B’s** posts/comments should not appear.

### Troubleshooting

- **`blockNotificationEmailSent: false`**: Check `RESEND_API_KEY`, `BLOCK_USER_EMAIL_ENABLED`, and that User B has `email` in Firebase Auth.
- **Resend error in logs**: Domain/sender not verified — use Resend dashboard to verify domain or use their test sender rules.
- **Privacy**: The email text is intentionally generic and does not name the blocker.
