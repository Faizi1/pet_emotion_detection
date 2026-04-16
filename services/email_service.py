"""
Transactional email via Resend (https://resend.com) — free tier suitable for low volume.

Set in environment:
  RESEND_API_KEY=re_...
  EMAIL_FROM=PetMood <onboarding@resend.dev>   # or your verified domain after DNS setup
  BLOCK_USER_EMAIL_ENABLED=1                   # set 0 to disable (block API still works)

If RESEND_API_KEY is missing, block notifications are skipped (logged) and the API still succeeds.
"""
import logging
import os
from typing import Optional, Tuple

import requests

logger = logging.getLogger(__name__)

RESEND_API_URL = "https://api.resend.com/emails"


def _resend_send(to_email: str, subject: str, html_body: str) -> Tuple[bool, Optional[str]]:
    api_key = (os.getenv("RESEND_API_KEY") or "").strip()
    if not api_key:
        return False, "RESEND_API_KEY not set"

    from_addr = (os.getenv("EMAIL_FROM") or "PetMood <onboarding@resend.dev>").strip()
    enabled = os.getenv("BLOCK_USER_EMAIL_ENABLED", "1").strip().lower() in ("1", "true", "yes")
    if not enabled:
        return False, "BLOCK_USER_EMAIL_ENABLED is off"

    try:
        resp = requests.post(
            RESEND_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "from": from_addr,
                "to": [to_email],
                "subject": subject,
                "html": html_body,
            },
            timeout=20,
        )
        if resp.status_code in (200, 201):
            return True, None
        return False, f"Resend HTTP {resp.status_code}: {resp.text[:500]}"
    except Exception as exc:  # pragma: no cover
        return False, str(exc)


def send_blocked_user_email(to_email: str, app_name: str = "PetMood") -> bool:
    """
    Notify the blocked user with a neutral message (no other user's identity).
    Returns True if Resend accepted the send, False otherwise.
    """
    if not to_email or "@" not in to_email:
        return False

    subject = f"{app_name} — account notice"
    html = f"""
    <p>Hello,</p>
    <p>Following activity in the {app_name} community, some interactions with your account may be limited for other members per our community guidelines.</p>
    <p>If you believe this is a mistake, you can contact support through the app.</p>
    <p>Thank you,<br/>The {app_name} team</p>
    """
    ok, err = _resend_send(to_email, subject, html)
    if ok:
        logger.info("Block notification email queued/sent to %s", to_email[:3] + "***")
    else:
        logger.warning("Block notification email not sent: %s", err)
    return ok
