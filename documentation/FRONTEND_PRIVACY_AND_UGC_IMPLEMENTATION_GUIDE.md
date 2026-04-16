# Frontend Implementation Guide (App Store Compliance)

This guide covers what the iOS/Android frontend must implement to satisfy:

- **Guidelines 5.1.1(i) / 5.1.2(i)**: Privacy (data collection/use + permission before sending personal data to third-party AI)
- **Guideline 1.2**: Safety (UGC precautions: filtering, reporting, blocking)

All endpoints below assume your backend base path is:

- **Base URL**: `https://<your-domain>/api/services/`
- **Auth**: `Authorization: Bearer <FIREBASE_ID_TOKEN>` (required for all endpoints listed here)

---

## AI / Third-Party Data Sharing Consent (Required)

### What the app must do

Before calling the scan endpoint that sends an **image/audio** to third-party AI providers, the app must:

- **Disclose** what data will be sent
- **Identify** who it is sent to
- **Ask permission** (explicit opt-in)

The backend enforces this: if consent is missing, scans return **403** with the required provider + disclosure.

### Providers currently used by backend

- **Images**: `nyckel`
- **Audio**: `assemblyai`

### 1) Fetch disclosure + current consent status

**GET** `privacy/ai-consent`

**Response (example)**

```json
{
  "consent": {
    "granted": false,
    "providers": [],
    "grantedAt": null,
    "revokedAt": null
  },
  "disclosure": {
    "nyckel": {
      "providerName": "Nyckel",
      "dataSent": ["image_bytes"],
      "purpose": "Pet emotion detection from user-submitted images"
    },
    "assemblyai": {
      "providerName": "AssemblyAI",
      "dataSent": ["audio_bytes"],
      "purpose": "Pet emotion detection from user-submitted audio"
    }
  }
}
```

### 2) Show consent screen (UI requirements)

When user starts a scan (or at onboarding), show a consent screen with:

- **What is sent**: image/audio uploaded by the user (bytes)
- **Who receives it**: provider name(s)
- **Purpose**: emotion detection
- **Choice**: allow / don’t allow
- Optional: per-provider toggles (recommended)

Suggested copy (customize to your UI style):

- Title: “Allow AI Analysis?”
- Body: “To analyze your pet’s emotion, we send the photo/audio you upload to our AI providers (Nyckel / AssemblyAI). We do not send your password. You can change this anytime in Settings.”
- Buttons:
  - “Allow” (grants consent)
  - “Not now” (denies; user can still browse but cannot run scans that require AI)

### 3) Store consent (grant or revoke)

**POST** `privacy/ai-consent`

**Grant request**

```json
{
  "granted": true,
  "providers": ["nyckel", "assemblyai"]
}
```

**Revoke request**

```json
{
  "granted": false,
  "providers": []
}
```

**Response**

```json
{
  "success": true,
  "consent": {
    "granted": true,
    "providers": ["nyckel", "assemblyai"],
    "grantedAt": "2026-03-17T12:00:00Z",
    "revokedAt": null
  }
}
```

### 4) Scan flow (handle consent enforcement)

**POST** `scans` (already existing endpoint)

If consent is missing, backend returns **403**:

```json
{
  "detail": "User consent required before sending media to third-party AI provider.",
  "requiredProvider": "assemblyai",
  "disclosure": {
    "providerName": "AssemblyAI",
    "dataSent": ["audio_bytes"],
    "purpose": "Pet emotion detection from user-submitted audio"
  }
}
```

**Frontend behavior on 403**

- Show the consent UI immediately.
- Pre-select the `requiredProvider` toggle, and show its `disclosure`.
- If the user accepts, call `POST privacy/ai-consent` then retry `POST scans`.
- If the user declines, do not retry and show a friendly message like: “You can enable AI analysis in Settings anytime.”

### Recommended Settings screen

Add “Privacy” → “AI Analysis Consent”:

- Toggle: “Allow AI analysis”
- If ON, show provider toggles:
  - Nyckel (images)
  - AssemblyAI (audio)

Implementation details:

- Load state from `GET privacy/ai-consent`
- Save state via `POST privacy/ai-consent`

---

## UGC Safety (Community Posts / Comments)

Your app includes UGC (posts/comments), so it must provide:

- A **method for filtering** objectionable content
- A **flag/report mechanism**
- A **block mechanism** that immediately removes content from the user’s feed and notifies the developer
- A developer process to act on reports within 24h (supported via admin API)

### 1) Objectionable content filtering (client behavior)

Backend rejects some objectionable text on create:

- `POST community/posts/create` may return **400**:

```json
{ "detail": "Post content violates community guidelines." }
```

- `POST community/comments/create` may return **400**:

```json
{ "detail": "Comment content violates community guidelines." }
```

Frontend behavior:

- Show inline error: “Your message contains prohibited content.”
- Let user edit and resubmit.

### 2) Report (flag) content

Add “Report” actions in UI:

- On a post: “Report Post”
- On a comment: “Report Comment”

#### Report a post

**POST** `community/posts/<post_id>/report`

Body:

```json
{ "reason": "Hate speech / harassment / spam / nudity / violence / other" }
```

Response:

```json
{ "reported": true, "reportId": "<id>" }
```

#### Report a comment

**POST** `community/comments/<post_id>/<comment_id>/report`

Body:

```json
{ "reason": "Harassment" }
```

Response:

```json
{ "reported": true, "reportId": "<id>" }
```

Frontend behavior:

- Show a reason picker, then submit.
- After success: toast “Thanks, we’ll review this.”

### 3) Block abusive users (instant removal)

Add “Block user” action anywhere you show a user (post author / comment author).

**POST** `community/users/<target_uid>/block`

Body:

```json
{ "reason": "Harassment" }
```

Response:

```json
{ "blocked": true, "targetUid": "<uid>", "reportId": "<id>" }
```

Frontend behavior:

- Immediately remove the blocked user’s content from current UI state:
  - Remove their posts from the feed list
  - Remove their comments from any open post thread
- On next refresh, backend will also hide blocked users server-side in:
  - `GET community/posts`
  - `GET community/posts/<post_id>`
  - `GET community/posts/<post_id>/comments`

### 4) Moderation queue (developer/admin)

The developer must act on reports within 24 hours. The backend provides:

**GET** `admin/moderation/reports?status=new&limit=50`

Notes:

- Requires admin Firebase privileges (server checks `IsAdminFirebase`)
- Returns moderation reports for posts/comments and block events

If you have a web admin dashboard, use this endpoint to list pending reports.

### 5) Moderation action API (NEW - close reports + enforce penalties)

To fully satisfy the "act within 24 hours" requirement, admin can now take explicit action per report:

**POST** `admin/moderation/reports/<report_id>/action`

Body:

```json
{
  "action": "remove_content_and_suspend_user",
  "adminNote": "Removed for abusive harassment",
  "suspendReason": "UGC policy violation"
}
```

Supported `action` values:

- `mark_reviewed` (reviewed, no penalty)
- `dismiss` (invalid report)
- `remove_content` (delete offending post/comment from report target)
- `suspend_user` (set offender user account to suspended in Firestore)
- `remove_content_and_suspend_user` (both)

Response (example):

```json
{
  "reportId": "abc123",
  "action": "remove_content_and_suspend_user",
  "postRemoved": true,
  "commentRemoved": false,
  "userSuspended": true,
  "suspendedUserId": "offenderUid",
  "reportStatus": "resolved"
}
```

Notes:

- Endpoint requires admin Firebase privileges.
- This endpoint is typically used by internal admin dashboard, not end-user mobile app.

---

## App Review “App Review Information” (what to tell Apple)

In App Store Connect → App Review Information, include a short statement like:

- “The app sends only user-submitted media (photos/audio) to third-party AI providers (Nyckel / AssemblyAI) for pet emotion detection. The app presents a consent prompt before any such transfer, and users can revoke consent in Settings.”

If you do not use one of these providers in production, remove it from the app text and consent toggles.

---

## Quick Checklist for Frontend

- **AI Consent**
  - Call `GET privacy/ai-consent` on app start and/or before scan
  - Show consent screen before scan if not granted
  - Save via `POST privacy/ai-consent`
  - Handle `403` from `POST scans` by presenting consent UI and retrying

- **UGC Safety**
  - Handle `400` “violates community guidelines” on post/comment creation
  - Add “Report” actions for posts/comments
  - Add “Block user” action and instantly remove content from UI

---

## Complete API Test Plan (Postman)

Use this to validate end-to-end before sending app for review.

### A) Privacy consent + scans

1. `GET privacy/ai-consent` (expect disclosure for Nyckel and AssemblyAI)
2. `POST privacy/ai-consent` with denied:

```json
{ "granted": false, "providers": [] }
```

3. Call `POST scans` with image/audio:
   - Expect `403` + `requiredProvider`
4. Grant consent:

```json
{ "granted": true, "providers": ["nyckel", "assemblyai"] }
```

5. Retry `POST scans`:
   - Expect `201` success

### B) UGC filters

1. `POST community/posts/create` with objectionable text
   - Expect `400`
2. `POST community/comments/create` with objectionable text
   - Expect `400`

### C) Report + block

1. Create normal post/comment
2. Report post (`POST community/posts/<post_id>/report`) -> expect `reported: true`
3. Report comment (`POST community/comments/<post_id>/<comment_id>/report`) -> expect `reported: true`
4. Block user (`POST community/users/<target_uid>/block`) -> expect `blocked: true`
5. Refresh feed/comments -> blocked user content hidden

### D) Admin moderation actions

1. Get queue: `GET admin/moderation/reports?status=new`
2. Pick `reportId`
3. Resolve with:

```json
{
  "action": "remove_content_and_suspend_user",
  "adminNote": "Policy violation confirmed",
  "suspendReason": "Abusive content"
}
```

4. Verify:
   - Report status changed to resolved/dismissed
   - Content removed if action included `remove_content`
   - User doc has `isSuspended: true` if action included `suspend_user`

---

## Ready message to send frontend developer

Copy-paste this:

> Backend is updated for Apple privacy + UGC compliance.  
> Please integrate using `FRONTEND_PRIVACY_AND_UGC_IMPLEMENTATION_GUIDE.md`.  
> Key tasks:
> 1. Add consent flow (`GET/POST privacy/ai-consent`) before scans.  
> 2. If scan returns 403, show consent modal and retry after grant.  
> 3. Add report actions for post/comment and block-user action with immediate UI removal.  
> 4. Handle 400 content-filter errors on post/comment create.  
> 5. For internal admin dashboard, integrate moderation queue + action endpoint to resolve reports within 24h.

