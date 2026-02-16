from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny

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
    signed_payload = request.data.get("signedPayload")
    if not signed_payload:
        return Response({"detail": "signedPayload missing"}, status=status.HTTP_400_BAD_REQUEST)

    client = AppleAppStoreClient()
    try:
        payload = client.decode_and_verify_jws(signed_payload)
    except Exception as exc:
        return Response({"detail": f"Invalid signedPayload: {exc}"}, status=status.HTTP_400_BAD_REQUEST)

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
    return Response({"received": True, "notificationType": notification_type, "subtype": subtype})


@api_view(["POST"])
def verify_receipt(request):
    """
    POST /api/subscriptions/verify-receipt
    Body: { receipt_data?, product_id, transaction_id, original_transaction_id? }
    """
    ser = VerifyReceiptSerializer(data=request.data)
    ser.is_valid(raise_exception=True)
    data = ser.validated_data

    uid = getattr(request.user, "uid", None)
    if not uid:
        return Response({"success": False, "message": "Unauthorized"}, status=status.HTTP_401_UNAUTHORIZED)

    product_id = data["product_id"]
    transaction_id = data["transaction_id"]

    if product_id not in PRODUCT_MAPPING:
        return Response({"success": False, "message": "Unknown product_id"}, status=status.HTTP_400_BAD_REQUEST)

    try:
        apple = AppleAppStoreClient()
        tx = apple.get_transaction_info(transaction_id)
    except AppleAppStoreServerAPIError as e:
        return Response({"success": False, "message": str(e)}, status=status.HTTP_400_BAD_REQUEST)
    except Exception:
        return Response({"success": False, "message": "Receipt verification failed"}, status=status.HTTP_400_BAD_REQUEST)

    if tx.product_id and tx.product_id != product_id:
        return Response({"success": False, "message": "product_id mismatch"}, status=status.HTTP_400_BAD_REQUEST)

    plan_type, period = PRODUCT_MAPPING[product_id]
    expires_at = tx.expires_at
    if not expires_at:
        return Response({"success": False, "message": "Missing expires_at"}, status=status.HTTP_400_BAD_REQUEST)

    # expires_at is timezone-aware UTC from Apple client
    is_active = True  # if expires_at exists, we still calculate active below using ISO parsing on reads
    payload = save_subscription_for_uid(
        uid=uid,
        product_id=product_id,
        plan_type=plan_type,
        period=period,
        transaction_id=tx.transaction_id,
        original_transaction_id=tx.original_transaction_id or data.get("original_transaction_id") or None,
        expires_at=expires_at,
        is_active=is_active,
    )

    return Response(
        {
            "success": True,
            "subscription": {
                "plan_type": plan_type,
                "period": period,
                "expires_at": payload["expires_at"],
                "is_active": True,
            },
        }
    )


@api_view(["GET"])
@permission_classes([IsAdminFirebase])
def admin_list_subscriptions(request):
    """
    GET /api/subscriptions/admin/list?uid=<firebase_uid>

    Admin/debug endpoint to list all subscriptions for a given uid.
    Requires Firebase admin privileges (IsAdminFirebase).
    """
    uid = request.query_params.get("uid") or getattr(request.user, "uid", None)
    if not uid:
        return Response({"detail": "uid is required"}, status=status.HTTP_400_BAD_REQUEST)

    subs = list_all_subscriptions(uid)
    return Response({"uid": uid, "subscriptions": subs})


@api_view(["GET"])
def subscription_status(request):
    """
    GET /api/subscriptions/status
    """
    uid = getattr(request.user, "uid", None)
    if not uid:
        return Response({"subscription": None}, status=status.HTTP_200_OK)

    sub = get_active_subscription(uid)
    if not sub:
        return Response({"subscription": None}, status=status.HTTP_200_OK)

    return Response(
        {
            "subscription": {
                "plan_type": sub.get("plan_type"),
                "period": sub.get("period"),
                "expires_at": sub.get("expires_at"),
                "is_active": True,
                "product_id": sub.get("product_id"),
            }
        }
    )


@api_view(["POST"])
def restore_purchases(request):
    """
    POST /api/subscriptions/restore
    Returns active subscriptions in DB for this uid.
    """
    uid = getattr(request.user, "uid", None)
    if not uid:
        return Response({"subscriptions": []}, status=status.HTTP_200_OK)

    subs = list_active_subscriptions(uid)

    return Response(
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
        }
    )


@api_view(["GET"])
@permission_classes([AllowAny])
def list_plans(request):
    """
    GET /api/subscriptions/plans

    Returns the list of available subscription plans derived from PRODUCT_MAPPING.
    This is intended for the client app's pricing/subscription screen.
    """
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
            }
        )
    return Response({"plans": data})



