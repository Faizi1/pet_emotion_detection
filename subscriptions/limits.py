"""
Subscription quota limits and enforcement helpers.

Free trial: max 2 profiles, 7 scans/day (account-wide).
Family plan: max 5 profiles, 7 scans/day (account-wide).
Daily scan counters reset at UTC midnight.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

from google.api_core.exceptions import FailedPrecondition

from services.firebase import get_firestore

TRIAL_MAX_PROFILES = 2
FAMILY_MAX_PROFILES = 5
TRIAL_SCANS_PER_DAY = 7
FAMILY_SCANS_PER_DAY = 7


def _pets_collection(uid: str):
    return get_firestore().collection("users").document(uid).collection("pets")


def _logs_collection(uid: str):
    return get_firestore().collection("users").document(uid).collection("emotion_logs")


def next_utc_midnight() -> datetime:
    start_of_day = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return start_of_day + timedelta(days=1)


def today_scan_count(uid: str) -> int:
    """Count today's scans in UTC for quota enforcement (account-wide)."""
    start_of_day = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day = start_of_day + timedelta(days=1)
    logs_ref = _logs_collection(uid)
    try:
        return len(
            list(
                logs_ref.where("timestamp", ">=", start_of_day)
                .where("timestamp", "<", end_of_day)
                .stream()
            )
        )
    except FailedPrecondition:
        logs = list(logs_ref.stream())
        count = 0
        for doc in logs:
            ts = (doc.to_dict() or {}).get("timestamp")
            if isinstance(ts, datetime) and start_of_day <= ts < end_of_day:
                count += 1
        return count


def pet_profile_count(uid: str) -> int:
    """Count existing pet profiles for the user."""
    return len(list(_pets_collection(uid).stream()))


def resolve_entitlement(active_sub: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Map an active subscription document to enforced quota limits.

    Trial takes precedence over plan type (family trial users get trial profile cap).
    Premium (paid, non-trial) has no profile or scan caps.
    """
    if not active_sub:
        return {
            "tier": "none",
            "max_profiles": None,
            "scans_per_day": None,
        }

    is_trial = bool(active_sub.get("is_trial"))
    plan_type = active_sub.get("plan_type") or "premium"

    if is_trial:
        return {
            "tier": "trial",
            "max_profiles": TRIAL_MAX_PROFILES,
            "scans_per_day": TRIAL_SCANS_PER_DAY,
        }

    if plan_type == "family":
        return {
            "tier": "family",
            "max_profiles": FAMILY_MAX_PROFILES,
            "scans_per_day": FAMILY_SCANS_PER_DAY,
        }

    return {
        "tier": plan_type,
        "max_profiles": None,
        "scans_per_day": None,
    }


def build_quota_payload(uid: str, active_sub: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Build quota usage snapshot for API responses."""
    entitlement = resolve_entitlement(active_sub)
    profiles_used = pet_profile_count(uid)
    scans_used = today_scan_count(uid)

    max_profiles = entitlement["max_profiles"]
    scans_per_day = entitlement["scans_per_day"]

    payload: Dict[str, Any] = {
        "tier": entitlement["tier"],
        "maxProfiles": max_profiles,
        "profilesUsed": profiles_used,
        "profilesRemaining": None if max_profiles is None else max(0, max_profiles - profiles_used),
        "scansPerDay": scans_per_day,
        "scansUsedToday": scans_used,
        "scansRemainingToday": None if scans_per_day is None else max(0, scans_per_day - scans_used),
        "resetsAt": next_utc_midnight().isoformat().replace("+00:00", "Z"),
    }
    return payload


def check_scan_allowed(
    uid: str, active_sub: Optional[Dict[str, Any]]
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Returns (allowed, error_body).
    error_body is suitable for a 429 response when not allowed.
    """
    entitlement = resolve_entitlement(active_sub)
    scans_per_day = entitlement["scans_per_day"]
    if scans_per_day is None:
        return True, None

    used_today = today_scan_count(uid)
    if used_today >= scans_per_day:
        return False, {
            "detail": "Daily scan limit reached",
            "limit": scans_per_day,
            "used": used_today,
            "remaining": 0,
            "resetsAt": next_utc_midnight().isoformat().replace("+00:00", "Z"),
            "tier": entitlement["tier"],
        }
    return True, None


def check_profile_creation_allowed(
    uid: str, active_sub: Optional[Dict[str, Any]]
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Returns (allowed, error_body).
    error_body is suitable for a 403 response when not allowed.
    """
    entitlement = resolve_entitlement(active_sub)
    max_profiles = entitlement["max_profiles"]
    if max_profiles is None:
        return True, None

    used = pet_profile_count(uid)
    if used >= max_profiles:
        return False, {
            "detail": "Profile limit reached for your subscription",
            "limit": max_profiles,
            "used": used,
            "remaining": 0,
            "tier": entitlement["tier"],
        }
    return True, None
