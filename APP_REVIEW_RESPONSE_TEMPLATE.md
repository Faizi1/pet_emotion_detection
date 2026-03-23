# App Store Review Response Template

Use this template in **App Store Connect -> Resolution Center** and **App Review Information**.
Replace placeholders like `<APP_NAME>` and `<TEST_ACCOUNT_EMAIL>`.

---

## 1) Resolution Center Reply (Privacy + UGC)

Hello App Review Team,

Thank you for your feedback. We have updated `<APP_NAME>` to address:

- Guideline 5.1.1(i) - Legal - Privacy - Data Collection
- Guideline 5.1.2(i) - Legal - Privacy - Data Use
- Guideline 1.2 - Safety - User-Generated Content

### A. Privacy and Third-Party AI Data Sharing (5.1.1(i), 5.1.2(i))

We now require explicit user consent before sending user-submitted media to third-party AI services.

What is sent:
- User-submitted image/audio bytes only (for pet emotion analysis)

Who data is sent to:
- Nyckel (image analysis)
- AssemblyAI (audio analysis)

What we changed:
- Added in-app disclosure and consent flow before AI analysis
- Added backend enforcement so scans are blocked if consent is missing
- Added user control to revoke consent in app settings

Technical implementation:
- `GET /api/services/privacy/ai-consent` -> returns disclosure + current consent status
- `POST /api/services/privacy/ai-consent` -> grants/revokes consent
- `POST /api/services/scans` -> returns 403 if consent is not granted for required provider

### B. UGC Safety Controls (1.2)

We implemented all required precautions:

1. Filtering objectionable content:
- Backend rejects objectionable post/comment text on create

2. Reporting objectionable content:
- `POST /api/services/community/posts/<post_id>/report`
- `POST /api/services/community/comments/<post_id>/<comment_id>/report`

3. Blocking abusive users:
- `POST /api/services/community/users/<target_uid>/block`
- Blocked user content is removed from blocker’s feed/comments immediately
- Block action is logged for moderation review

4. Developer moderation within 24 hours:
- Moderation queue: `GET /api/services/admin/moderation/reports`
- Moderation actions:
  - `POST /api/services/admin/moderation/reports/<report_id>/action`
  - Supports remove content / suspend user / resolve or dismiss reports

We confirm we review and act on moderation reports within 24 hours.

Thank you.

---

## 2) App Review Information (Short Form)

You can paste this in **App Review Information -> Notes**:

`<APP_NAME>` sends only user-submitted media (photo/audio) to third-party AI providers (Nyckel for images, AssemblyAI for audio) for pet emotion analysis. The app clearly discloses what is sent and to whom, and requests user permission before any transfer. Users can revoke consent in app settings.  
The app also includes UGC safety features: objectionable content filtering, report/flag actions, user blocking with immediate feed removal, and a moderation workflow where reports are reviewed and actioned within 24 hours.

---

## 3) Reviewer Test Steps (Optional but recommended)

If needed, provide this quick flow:

1. Sign in with:
- Email: `<TEST_ACCOUNT_EMAIL>`
- Password: `<TEST_ACCOUNT_PASSWORD>`

2. Privacy consent check:
- Start an image/audio scan
- App shows AI disclosure + consent screen
- If consent denied, scan is blocked
- If consent granted, scan succeeds

3. UGC checks:
- Create post/comment with prohibited text -> rejected
- Report a post/comment -> accepted
- Block a user -> their content disappears from feed/comments

4. Moderation handling:
- Reports are available in admin moderation queue and can be actioned (remove content / suspend user)

---

## 4) Privacy Policy Text Checklist

Ensure your Privacy Policy explicitly states:

- What data is collected (including user-submitted media)
- How data is collected (user uploads)
- Why data is used (pet emotion analysis, app features)
- Third-party processors used (Nyckel, AssemblyAI)
- That equivalent protection is required from third parties
- User controls (consent and revocation)

