# Support Feature Workflow (PetMood)

This document explains the complete support flow after the latest fixes.

## 1) Mobile App Behavior (User Side)

### Endpoint
- `POST /api/support/send`

### Request body
```json
{
  "email": "user@example.com",
  "details": "I have an issue with ..."
}
```

### What happens on submit
1. Backend validates `email` and `details`.
2. Backend creates a Firestore document in `support_messages` with:
   - `email`
   - `details`
   - `userId` (if authenticated)
   - `status = "new"`
   - `createdAt`, `updatedAt`
   - `adminReply = ""`
3. Backend returns success to app.
4. Backend tries to email support/admin recipients (non-blocking).

### Success response
```json
{
  "message": "Support message sent successfully. We will get back to you soon.",
  "supportId": "generated_firestore_id"
}
```

## 2) Admin Panel Behavior

### Support page
- URL: `/admin-panel/support/`
- Reads all documents from Firestore collection `support_messages`
- Supports filters:
  - email
  - status
  - priority

### Action buttons in table
- **View**: opens modal with real message details (no placeholder text).
- **Reply**: opens reply modal, saves reply + status.
- **Update Status**: opens status modal and updates only status.
- **Delete**: deletes support message.

### Admin routes (now implemented)
- `POST /admin-panel/support/<message_id>/reply/`
- `POST /admin-panel/support/<message_id>/update-status/`
- `POST /admin-panel/support/<message_id>/delete/`

If these routes return 404, backend is not running latest deployed code.

## 3) Status Values

### Primary statuses (current)
- `new`
- `read`
- `replied`

### Legacy statuses (still accepted for backward compatibility)
- `pending`
- `in_progress`
- `resolved`
- `closed`

## 4) Email Notifications

Email is sent using Resend integration in `services/email_service.py`.

### A) On new support submission (user -> support team)
- Trigger: `POST /api/support/send`
- Uses env var: `SUPPORT_NOTIFICATION_EMAILS`
- Supports comma-separated recipient list.
- Example:
  - `SUPPORT_NOTIFICATION_EMAILS=owner@petmood.com,support@petmood.com`

### B) On admin reply (support team -> user)
- Trigger: admin reply action
- Sends email to the message submitter email.

### Required env vars
- `RESEND_API_KEY`
- `EMAIL_FROM` (recommended)
- `SUPPORT_NOTIFICATION_EMAILS` (for admin/team notification)
- `BLOCK_USER_EMAIL_ENABLED=1` (email sending gate used by current helper)

## 5) Firestore Data Model (Support)

Collection: `support_messages`

Document example:
```json
{
  "email": "user@example.com",
  "details": "My issue text",
  "userId": "firebase_uid_or_null",
  "status": "new",
  "adminReply": "",
  "createdAt": "timestamp",
  "updatedAt": "timestamp",
  "priority": "optional"
}
```

## 6) Troubleshooting

### Problem: old modal text still appears
Symptom:
- "Message details for ID ..."
- "This would show full message content ..."

Cause:
- Old frontend template is still deployed.

Fix:
1. Redeploy backend.
2. Hard refresh browser (`Ctrl+F5`).
3. Reopen `/admin-panel/support/`.

### Problem: update status gives 404
Cause:
- Old backend without new admin routes.

Fix:
1. Deploy latest backend code.
2. Verify route URL starts with `/admin-panel/support/<id>/update-status/`.
3. Ensure method is POST from modal form.

### Problem: reply saves but no email received
Checklist:
1. `RESEND_API_KEY` is set.
2. `EMAIL_FROM` is valid/verified.
3. Recipient inbox not in spam.
4. App logs: check warning messages from `email_service.py`.

### Problem: user submits support but team email not received
Checklist:
1. `SUPPORT_NOTIFICATION_EMAILS` is set with valid emails.
2. `RESEND_API_KEY` is set.
3. `BLOCK_USER_EMAIL_ENABLED=1`.
4. Check backend logs for Resend errors.

## 7) Quick Test Plan

1. Submit support from app with test email and text.
2. Confirm new record appears at `/admin-panel/support/`.
3. Click **View** and verify full details are shown.
4. Click **Update Status**, set to `read`, submit, verify status badge changes.
5. Click **Reply**, add reply text, submit, verify:
   - `adminReply` stored
   - status updated
   - reply email delivered to user
6. Confirm support-team inbox received new-message notification email from step 1.

