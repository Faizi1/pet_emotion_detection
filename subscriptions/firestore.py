from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from services.firebase import get_firestore


SUBSCRIPTIONS_SUBCOLLECTION = "subscriptions"


def _iso_z(dt: datetime) -> str:
    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc.isoformat().replace("+00:00", "Z")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def subscriptions_collection(uid: str):
    return get_firestore().collection("users").document(uid).collection(SUBSCRIPTIONS_SUBCOLLECTION)


def save_subscription_for_uid(
    *,
    uid: str,
    product_id: str,
    plan_type: str,
    period: str,
    transaction_id: str,
    original_transaction_id: Optional[str],
    expires_at: datetime,
    is_active: bool,
) -> Dict[str, Any]:
    """
    Store subscription under: users/{uid}/subscriptions/{transaction_id}
    """
    doc = subscriptions_collection(uid).document(transaction_id)
    payload = {
        "uid": uid,
        "product_id": product_id,
        "plan_type": plan_type,
        "period": period,
        "transaction_id": transaction_id,
        "original_transaction_id": original_transaction_id,
        "expires_at": _iso_z(expires_at),
        "is_active": bool(is_active),
        "updated_at": _iso_z(_now_utc()),
    }
    # Preserve created_at if already exists
    existing = doc.get()
    if existing.exists:
        prev = existing.to_dict() or {}
        payload["created_at"] = prev.get("created_at") or payload["updated_at"]
    else:
        payload["created_at"] = payload["updated_at"]
    doc.set(payload, merge=True)
    return payload


def get_active_subscription(uid: str) -> Optional[Dict[str, Any]]:
    """
    Returns the newest active subscription (by expires_at string).
    """
    now = _now_utc()
    # Firestore can order by string ISO timestamps lexicographically if same format (ISO Z).
    docs = (
        subscriptions_collection(uid)
        .where("is_active", "==", True)
        .order_by("expires_at", direction="DESCENDING")
        .limit(5)
        .stream()
    )
    best: Optional[Dict[str, Any]] = None
    for d in docs:
        data = d.to_dict() or {}
        expires_at = data.get("expires_at")
        try:
            dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00")) if expires_at else None
        except Exception:
            dt = None
        if dt and dt > now:
            best = data
            break
    return best


def list_active_subscriptions(uid: str) -> List[Dict[str, Any]]:
    now = _now_utc()
    docs = (
        subscriptions_collection(uid)
        .where("is_active", "==", True)
        .order_by("expires_at", direction="DESCENDING")
        .stream()
    )
    out: List[Dict[str, Any]] = []
    for d in docs:
        data = d.to_dict() or {}
        expires_at = data.get("expires_at")
        try:
            dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00")) if expires_at else None
        except Exception:
            dt = None
        if dt and dt > now:
            out.append(data)
    return out


def list_all_subscriptions(uid: str) -> List[Dict[str, Any]]:
    """
    Admin/debug helper: returns all subscription docs for a uid, newest first.
    """
    docs = (
        subscriptions_collection(uid)
        .order_by("expires_at", direction="DESCENDING")
        .stream()
    )
    out: List[Dict[str, Any]] = []
    for d in docs:
        data = d.to_dict() or {}
        out.append(data)
    return out


