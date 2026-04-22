# Firebase Login Persistence (Frontend Note)

## Purpose

Keep users logged in without backend refresh-token endpoint by using Firebase SDK session/token handling correctly on mobile frontend.

## Backend impact

No new backend implementation is required.

- Existing backend token verification is enough.
- Backend still expects valid Firebase ID token in `Authorization: Bearer <idToken>`.
- ID tokens expire around every 1 hour by design.

## Frontend required implementation

1. Use Firebase Auth session persistence (default behavior in mobile SDK).
2. Do not reuse one old ID token forever from storage.
3. Before protected API calls, get current token from Firebase user:
   - `currentUser.getIdToken()`
4. If API returns auth failure due to token expiry:
   - call `currentUser.getIdToken(true)` to force refresh
   - retry API request once with new token
5. Logout user only when refresh/retry fails.

## Expected result

- User stays logged in long-term (days/weeks) until explicit logout or app data/session is cleared.
- No forced logout every 1 hour.

## Frontend quick checklist

- [ ] Token fetched from Firebase before API call
- [ ] One-time retry on expired token with forced refresh
- [ ] No permanent cached/stale token usage
- [ ] Logout only after refresh failure
