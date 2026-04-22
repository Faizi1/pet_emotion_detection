from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
import json
from datetime import datetime, timezone

from services.permissions import IsAdminFirebase
from .apple import AppleAppStoreClient, AppleAppStoreServerAPIError
from .serializers import VerifyReceiptSerializer
from .firestore import (
    get_active_subscription,
    get_latest_subscription,
    list_active_subscriptions,
    save_subscription_for_uid,
    list_all_subscriptions,
)


PRODUCT_MAPPING = {
    "com.petmood.premium.monthly": ("premium", "monthly"),
    "com.petmood.premium.annual": ("premium", "annual"),
    "com.petmood.family.monthly": ("family", "monthly"),
    "com.petmood.family.annual": ("family", "annual"),
}

# 7-day introductory trial metadata for UI guidance.
# NOTE: Real trial eligibility/billing behavior is controlled by App Store Connect + StoreKit.
TRIAL_DAYS = 7
TRIAL_ELIGIBLE_PRODUCTS = {
    "com.petmood.premium.monthly",
    "com.petmood.family.monthly",
}


def _debug_request(api_name: str, request) -> None:
    try:
        payload = request.data if request.method in {"POST", "PUT", "PATCH"} else request.query_params.dict()
    except Exception:
        payload = {}
    print(f"[subscriptions][{api_name}] request_payload={json.dumps(payload, default=str)}")


def _debug_response(api_name: str, body, status_code: int = status.HTTP_200_OK) -> Response:
    print(
        f"[subscriptions][{api_name}] response_status={status_code} "
        f"response_body={json.dumps(body, default=str)}"
    )
    return Response(body, status=status_code)


def _debug_payload(api_name: str, label: str, payload) -> None:
    print(f"[subscriptions][{api_name}] {label}={json.dumps(payload, default=str)}")


def _parse_iso_utc(value: str):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _trial_days_left(expires_at_iso: str) -> int:
    expires_at = _parse_iso_utc(expires_at_iso)
    if not expires_at:
        return 0
    now = datetime.now(timezone.utc)
    if expires_at <= now:
        return 0
    seconds_left = (expires_at - now).total_seconds()
    # Ceil to communicate whole remaining trial days to client UI.
    return int((seconds_left + 86399) // 86400)


def _subscription_response_shape(sub: dict, *, is_active: bool) -> dict:
    expires_at = sub.get("expires_at")
    is_trial = bool(sub.get("is_trial", False))
    return {
        "plan_type": sub.get("plan_type"),
        "period": sub.get("period"),
        "expires_at": expires_at,
        "is_active": bool(is_active),
        "product_id": sub.get("product_id"),
        "is_trial": is_trial,
        "trial_days": int(sub.get("trial_days") or 0),
        "trial_days_left": _trial_days_left(expires_at) if is_trial else 0,
        "offer_type": sub.get("offer_type"),
        "offer_period": sub.get("offer_period"),
        "environment": sub.get("environment"),
    }


@api_view(["POST"])
@permission_classes([AllowAny])  # Apple server, no Firebase token
def app_store_webhook(request):
    """
    POST /api/subscriptions/webhook

    App Store Server Notifications v2 endpoint.
    Expects JSON body: { "signedPayload": "<JWS>" }
   
    NOTE: To fully link notifications to your Firebase users, the iOS app
    should pass the Firebase uid as `appAccountToken` in the original purchase.
    """
    _debug_request("app_store_webhook", request)
    signed_payload = request.data.get("signedPayload")
    if not signed_payload:
        return _debug_response("app_store_webhook", {"detail": "signedPayload missing"}, status.HTTP_400_BAD_REQUEST)

    client = AppleAppStoreClient()
    decoded_notification_payload = {}
    try:
        payload = client.decode_and_verify_jws(signed_payload)
        decoded_notification_payload = payload
        _debug_payload("app_store_webhook", "decoded_notification_payload", payload)
    except Exception as exc:
        return _debug_response(
            "app_store_webhook",
            {"detail": f"Invalid signedPayload: {exc}"},
            status.HTTP_400_BAD_REQUEST,
        )

    # Basic fields from notification
    notification_type = payload.get("notificationType")
    subtype = payload.get("subtype")
    data = payload.get("data") or {}

    # Try to decode signedTransactionInfo to update subscription if we can map to a user.
    signed_tx = data.get("signedTransactionInfo")
    app_account_token = None
    tx_product_id = None
    tx_expires_at = None
    tx_transaction_id = None
    tx_payload = {}

    if signed_tx:
        try:
            tx_payload = client.decode_and_verify_jws(signed_tx)
            _debug_payload("app_store_webhook", "decoded_transaction_payload", tx_payload)
            app_account_token = tx_payload.get("appAccountToken")
            tx_product_id = tx_payload.get("productId")
            tx_transaction_id = tx_payload.get("transactionId")
            expires_ms = tx_payload.get("expiresDate")

            if expires_ms:
                tx_expires_at = datetime.fromtimestamp(expires_ms / 1000.0, tz=timezone.utc)
        except Exception:
            # If decoding fails, we still return 200 so Apple doesn't keep retrying.
            _debug_payload("app_store_webhook", "decoded_transaction_payload_error", {"detail": "signedTransactionInfo decode failed"})
            pass

    # If appAccountToken is present and we can map to Firebase uid, persist subscription.
    if app_account_token and tx_product_id and tx_transaction_id and tx_expires_at:
        uid = app_account_token  # convention: appAccountToken == Firebase uid
        plan = PRODUCT_MAPPING.get(tx_product_id)
        if plan:
            plan_type, period = plan
            save_subscription_for_uid(
                uid=uid,
                product_id=tx_product_id,
                plan_type=plan_type,
                period=period,
                transaction_id=tx_transaction_id,
                original_transaction_id=tx_payload.get("originalTransactionId"),
                expires_at=tx_expires_at,
                is_active=True,
                is_trial=bool((tx_payload.get("offerType") or 0) == 1 or tx_payload.get("offerDiscountType") == "FREE_TRIAL"),
                trial_days=TRIAL_DAYS if tx_product_id in TRIAL_ELIGIBLE_PRODUCTS else 0,
                offer_type=str(tx_payload.get("offerType") or ""),
                offer_period=tx_payload.get("offerPeriod"),
                environment=tx_payload.get("environment"),
            )

    # Always respond 200 OK so Apple knows we handled it.
    return _debug_response(
        "app_store_webhook",
        {
            "received": True,
            "notificationType": notification_type,
            "subtype": subtype,
            "decodedNotificationPayload": decoded_notification_payload,
            "decodedTransactionPayload": tx_payload,
        },
    )


@api_view(["POST"])
def verify_receipt(request):
    """
    POST /api/subscriptions/verify-receipt
    Body: { receipt_data?, product_id, transaction_id, original_transaction_id?, signed_transaction_jws? }
    """
    _debug_request("verify_receipt", request)
    ser = VerifyReceiptSerializer(data=request.data)
    ser.is_valid(raise_exception=True)
    data = ser.validated_data

    uid = getattr(request.user, "uid", None)
    if not uid:
        return _debug_response("verify_receipt", {"success": False, "message": "Unauthorized"}, status.HTTP_401_UNAUTHORIZED)

    product_id = data["product_id"]
    transaction_id = data["transaction_id"]

    if product_id not in PRODUCT_MAPPING:
        return _debug_response(
            "verify_receipt",
            {"success": False, "message": "Unknown product_id"},
            status.HTTP_400_BAD_REQUEST,
        )

    signed_jws = data.get("signed_transaction_jws")
    plan_type, period = PRODUCT_MAPPING[product_id]
    expires_at = None
    jws_payload = {}
    final_transaction_id = transaction_id
    final_original_transaction_id = data.get("original_transaction_id") or None

    try:
        apple = AppleAppStoreClient()

        # If StoreKit 2 JWS is provided, validate and use it first.
        if signed_jws:
            jws_payload = apple.decode_and_verify_jws(signed_jws)
            _debug_payload("verify_receipt", "decoded_signed_transaction_jws", jws_payload)

            jws_bundle_id = jws_payload.get("bundleId")
            if jws_bundle_id and jws_bundle_id != apple.bundle_id:
                return _debug_response(
                    "verify_receipt",
                    {"success": False, "message": "bundle_id mismatch"},
                    status.HTTP_400_BAD_REQUEST,
                )

            jws_product_id = jws_payload.get("productId")
            if jws_product_id and jws_product_id != product_id:
                return _debug_response(
                    "verify_receipt",
                    {"success": False, "message": "product_id mismatch"},
                    status.HTTP_400_BAD_REQUEST,
                )

            jws_tx_id = str(jws_payload.get("transactionId") or "")
            if jws_tx_id and jws_tx_id != str(transaction_id):
                return _debug_response(
                    "verify_receipt",
                    {"success": False, "message": "transaction_id mismatch"},
                    status.HTTP_400_BAD_REQUEST,
                )

            expires_ms = jws_payload.get("expiresDate")
            if expires_ms:
                expires_at = datetime.fromtimestamp(expires_ms / 1000.0, tz=timezone.utc)
            final_transaction_id = jws_tx_id or final_transaction_id
            final_original_transaction_id = (
                jws_payload.get("originalTransactionId")
                or final_original_transaction_id
            )

        # If we still don't have expiry from JWS, fetch authoritative transaction info.
        if not expires_at:
            tx = apple.get_transaction_info(transaction_id)
            if tx.product_id and tx.product_id != product_id:
                return _debug_response(
                    "verify_receipt",
                    {"success": False, "message": "product_id mismatch"},
                    status.HTTP_400_BAD_REQUEST,
                )
            expires_at = tx.expires_at
            final_transaction_id = tx.transaction_id or final_transaction_id
            final_original_transaction_id = tx.original_transaction_id or final_original_transaction_id

    except AppleAppStoreServerAPIError as e:
        return _debug_response("verify_receipt", {"success": False, "message": str(e)}, status.HTTP_400_BAD_REQUEST)
    except Exception:
        return _debug_response(
            "verify_receipt",
            {"success": False, "message": "Receipt verification failed"},
            status.HTTP_400_BAD_REQUEST,
        )

    if not expires_at:
        return _debug_response(
            "verify_receipt",
            {"success": False, "message": "Missing expires_at"},
            status.HTTP_400_BAD_REQUEST,
        )

    # expires_at is timezone-aware UTC from Apple client
    is_active = True  # if expires_at exists, we still calculate active below using ISO parsing on reads
    payload = save_subscription_for_uid(
        uid=uid,
        product_id=product_id,
        plan_type=plan_type,
        period=period,
        transaction_id=final_transaction_id,
        original_transaction_id=final_original_transaction_id,
        expires_at=expires_at,
        is_active=is_active,
        is_trial=bool((jws_payload.get("offerType") or 0) == 1 or jws_payload.get("offerDiscountType") == "FREE_TRIAL"),
        trial_days=TRIAL_DAYS if product_id in TRIAL_ELIGIBLE_PRODUCTS else 0,
        offer_type=str(jws_payload.get("offerType") or ""),
        offer_period=jws_payload.get("offerPeriod"),
        environment=jws_payload.get("environment"),
    )

    return _debug_response(
        "verify_receipt",
        {
            "success": True,
            "subscription": _subscription_response_shape(payload, is_active=True),
        },
    )


@api_view(["GET"])
@permission_classes([IsAdminFirebase])
def admin_list_subscriptions(request):
    """
    GET /api/subscriptions/admin/list?uid=<firebase_uid>

    Admin/debug endpoint to list all subscriptions for a given uid.
    Requires Firebase admin privileges (IsAdminFirebase).
    """
    _debug_request("admin_list_subscriptions", request)
    uid = request.query_params.get("uid") or getattr(request.user, "uid", None)
    if not uid:
        return _debug_response(
            "admin_list_subscriptions",
            {"detail": "uid is required"},
            status.HTTP_400_BAD_REQUEST,
        )

    subs = list_all_subscriptions(uid)
    return _debug_response("admin_list_subscriptions", {"uid": uid, "subscriptions": subs})


@api_view(["GET"])
def subscription_status(request):
    """
    GET /api/subscriptions/status
    """
    _debug_request("subscription_status", request)
    uid = getattr(request.user, "uid", None)
    if not uid:
        return _debug_response("subscription_status", {"subscription": None}, status.HTTP_200_OK)

    sub = get_active_subscription(uid)
    if sub:
        return _debug_response(
            "subscription_status",
            {"subscription": _subscription_response_shape(sub, is_active=True)},
            status.HTTP_200_OK,
        )

    # No active entitlement. Return last known subscription so app can show
    # "expired" state instead of ambiguous null.
    latest = get_latest_subscription(uid)
    if latest:
        return _debug_response(
            "subscription_status",
            {
                "subscription": _subscription_response_shape(latest, is_active=False),
                "access_active": False,
                "reason": "expired_or_not_renewed",
            },
            status.HTTP_200_OK,
        )

    return _debug_response("subscription_status", {"subscription": None}, status.HTTP_200_OK)


@api_view(["POST"])
def restore_purchases(request):
    """
    POST /api/subscriptions/restore
    Returns active subscriptions in DB for this uid.
    """
    _debug_request("restore_purchases", request)
    uid = getattr(request.user, "uid", None)
    if not uid:
        return _debug_response("restore_purchases", {"subscriptions": []}, status.HTTP_200_OK)

    subs = list_active_subscriptions(uid)

    return _debug_response(
        "restore_purchases",
        {
            "subscriptions": [
                _subscription_response_shape(s, is_active=True)
                for s in subs
            ]
        },
    )


@api_view(["POST"])
def cancel_subscription(request):
    """
    POST /api/subscriptions/cancel

    Apple subscriptions cannot be canceled directly by backend API.
    User must cancel from Apple subscription management UI.
    """
    _debug_request("cancel_subscription", request)
    uid = getattr(request.user, "uid", None)
    if not uid:
        return _debug_response("cancel_subscription", {"detail": "Unauthorized"}, status.HTTP_401_UNAUTHORIZED)

    active_sub = get_active_subscription(uid)
    latest_sub = active_sub or get_latest_subscription(uid)
    subscription_payload = (
        _subscription_response_shape(latest_sub, is_active=bool(active_sub))
        if latest_sub else None
    )

    manage_url = "https://apps.apple.com/account/subscriptions"
    return _debug_response(
        "cancel_subscription",
        {
            "success": True,
            "platform": "apple",
            "canCancelDirectlyFromBackend": False,
            "message": "Apple requires user-initiated cancellation from App Store subscription settings.",
            "manageSubscriptionUrl": manage_url,
            "subscription": subscription_payload,
        },
        status.HTTP_200_OK,
    )


@api_view(["GET"])
@permission_classes([AllowAny])
def list_plans(request):
    """
    GET /api/subscriptions/plans

    Returns the list of available subscription plans derived from PRODUCT_MAPPING.
    This is intended for the client app's pricing/subscription screen.
    """
    _debug_request("list_plans", request)
    data = []
    for product_id, (plan_type, period) in PRODUCT_MAPPING.items():
        name = f"{plan_type.capitalize()} {period.capitalize()}"
        data.append(
            {
                "product_id": product_id,
                "name": name,
                "plan_type": plan_type,
                "period": period,
                "platform": "apple",
                "price_display": "",
                "trial_offer": {
                    "enabled": product_id in TRIAL_ELIGIBLE_PRODUCTS,
                    "days": TRIAL_DAYS if product_id in TRIAL_ELIGIBLE_PRODUCTS else 0,
                    "type": "introductory",
                },
            }
        )
    return _debug_response("list_plans", {"plans": data})



