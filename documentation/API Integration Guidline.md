# Message to Frontend Developer

Hi, please integrate the following APIs for App Store compliance (privacy + UGC safety).  
Base URL: `{{BASE_URL}}/api/services`  
Auth header for protected APIs: `Authorization: Bearer {{FIREBASE_ID_TOKEN}}`

## 1) Privacy + AI consent (mandatory before scan)

### Step A: On app start (or before scan), call:
- `GET /privacy/ai-consent`

Use response to:
- Show what data is sent
- Show which provider receives data
- Determine if consent already granted

### Step B: Save user consent:
- `POST /privacy/ai-consent`

Examples:
- Grant:
```json
{ "granted": true, "providers": ["nyckel", "assemblyai"] }
```
- Revoke:
```json
{ "granted": false, "providers": [] }
```

### Step C: Scan flow handling:
- Existing endpoint: `POST /scans`
- If consent is missing, backend returns `403` with `requiredProvider` + `disclosure`

Required frontend behavior:
- On `403`, show consent modal immediately
- If user allows -> call `POST /privacy/ai-consent` -> retry scan
- If user denies -> stop scan and show clear message

---

## 2) UGC safety features

### A) Filtering (already enforced by backend)
- `POST /community/posts/create` and `POST /community/comments/create`
- May return `400` with:
  - `Post content violates community guidelines.`
  - `Comment content violates community guidelines.`

Frontend:
- Show inline error and allow user to edit/resubmit

### B) Report content
- Report post: `POST /community/posts/<post_id>/report`
- Report comment: `POST /community/comments/<post_id>/<comment_id>/report`

Frontend:
- Add "Report" action menu for posts/comments
- Show reason picker
- Show success toast after submit

### C) Block abusive users
- `POST /community/users/<target_uid>/block`

Frontend:
- Add "Block user" action on author profile/menu
- Immediately remove blocked user's posts/comments from local UI
- On refresh, backend also filters blocked user content

---

## 3) Internal admin moderation (for 24h enforcement)

### List reports:
- `GET /admin/moderation/reports?status=new&limit=50`

### Take action on a report:
- `POST /admin/moderation/reports/<report_id>/action`

Actions:
- `mark_reviewed`
- `dismiss`
- `remove_content`
- `suspend_user`
- `remove_content_and_suspend_user`

Note:
- These admin APIs require admin Firebase token and are for admin panel/internal tooling.

---

## 4) Minimum frontend checklist (must complete)

- [ ] Implement consent screen + settings toggle
- [ ] Handle scan 403 consent-required flow
- [ ] Add report post/comment actions
- [ ] Add block user action + immediate UI removal
- [ ] Handle objectionable content 400 errors
- [ ] (Admin UI) integrate moderation queue + moderation action endpoint

