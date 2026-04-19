from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
import json

from services.permissions import IsAdminFirebase
from .apple import AppleAppStoreClient, AppleAppStoreServerAPIError
from .serializers import VerifyReceiptSerializer
from .firestore import (
    get_active_subscription,
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
    try:
        payload = client.decode_and_verify_jws(signed_payload)
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

    if signed_tx:
        try:
            tx_payload = client.decode_and_verify_jws(signed_tx)
            app_account_token = tx_payload.get("appAccountToken")
            tx_product_id = tx_payload.get("productId")
            tx_transaction_id = tx_payload.get("transactionId")
            expires_ms = tx_payload.get("expiresDate")

            from datetime import datetime, timezone

            if expires_ms:
                tx_expires_at = datetime.fromtimestamp(expires_ms / 1000.0, tz=timezone.utc)
        except Exception:
            # If decoding fails, we still return 200 so Apple doesn't keep retrying.
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
            )

    # Always respond 200 OK so Apple knows we handled it.
    return _debug_response(
        "app_store_webhook",
        {"received": True, "notificationType": notification_type, "subtype": subtype},
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
    final_transaction_id = transaction_id
    final_original_transaction_id = data.get("original_transaction_id") or None

    try:
        apple = AppleAppStoreClient()

        # If StoreKit 2 JWS is provided, validate and use it first.
        if signed_jws:
            jws_payload = apple.decode_and_verify_jws(signed_jws)

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

            from datetime import datetime, timezone
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
    )

    return _debug_response(
        "verify_receipt",
        {
            "success": True,
            "subscription": {
                "plan_type": plan_type,
                "period": period,
                "expires_at": payload["expires_at"],
                "is_active": True,
            },
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
    if not sub:
        return _debug_response("subscription_status", {"subscription": None}, status.HTTP_200_OK)

    return _debug_response(
        "subscription_status",
        {
            "subscription": {
                "plan_type": sub.get("plan_type"),
                "period": sub.get("period"),
                "expires_at": sub.get("expires_at"),
                "is_active": True,
                "product_id": sub.get("product_id"),
            }
        },
    )


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
                {
                    "is_active": True,
                    "plan_type": s.get("plan_type"),
                    "period": s.get("period"),
                    "expires_at": s.get("expires_at"),
                    "product_id": s.get("product_id"),
                }
                for s in subs
            ]
        },
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



