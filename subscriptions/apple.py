from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import jwt
import requests
from django.conf import settings


@dataclass(frozen=True)
class AppleTransactionInfo:
    product_id: str
    transaction_id: str
    original_transaction_id: Optional[str]
    expires_at: Optional[datetime]


class AppleAppStoreServerAPIError(RuntimeError):
    pass


def _utc_from_ms(ms: Optional[int]) -> Optional[datetime]:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def _read_private_key() -> str:
    """
    Supports either:
      - settings.APPLE_IAP_PRIVATE_KEY (full PEM text)
      - settings.APPLE_IAP_PRIVATE_KEY_PATH (path to .p8)
    """
    key = getattr(settings, "APPLE_IAP_PRIVATE_KEY", "") or os.getenv("APPLE_IAP_PRIVATE_KEY", "")
    if key.strip():
        return key
    path = getattr(settings, "APPLE_IAP_PRIVATE_KEY_PATH", "") or os.getenv("APPLE_IAP_PRIVATE_KEY_PATH", "")
    if not path:
        raise AppleAppStoreServerAPIError("Missing APPLE_IAP_PRIVATE_KEY or APPLE_IAP_PRIVATE_KEY_PATH")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class AppleAppStoreClient:
    """
    Minimal App Store Server API client:
      - Auth via ES256 JWT (App Store Connect API Key)
      - Fetch transaction info by transactionId
      - Verify and decode JWS payload using Apple's keys endpoint
    """

    def __init__(self) -> None:
        self.key_id = getattr(settings, "APPLE_IAP_KEY_ID", "") or os.getenv("APPLE_IAP_KEY_ID", "")
        self.issuer_id = getattr(settings, "APPLE_IAP_ISSUER_ID", "") or os.getenv("APPLE_IAP_ISSUER_ID", "")
        self.bundle_id = getattr(settings, "APPLE_IAP_BUNDLE_ID", "") or os.getenv("APPLE_IAP_BUNDLE_ID", "")
        self.use_sandbox = bool(
            (getattr(settings, "APPLE_IAP_USE_SANDBOX", None))
            if getattr(settings, "APPLE_IAP_USE_SANDBOX", None) is not None
            else (os.getenv("APPLE_IAP_USE_SANDBOX", "0") in ("1", "true", "True"))
        )

        if not self.key_id or not self.issuer_id or not self.bundle_id:
            raise AppleAppStoreServerAPIError("Missing APPLE_IAP_KEY_ID / APPLE_IAP_ISSUER_ID / APPLE_IAP_BUNDLE_ID")

        self._private_key = _read_private_key()
        self._cached_keys: Optional[Dict[str, Any]] = None
        self._cached_keys_at: float = 0.0

    @property
    def base_url(self) -> str:
        return "https://api.storekit-sandbox.itunes.apple.com" if self.use_sandbox else "https://api.storekit.itunes.apple.com"

    def _generate_token(self) -> str:
        now = int(time.time())
        headers = {"alg": "ES256", "kid": self.key_id, "typ": "JWT"}
        payload = {
            "iss": self.issuer_id,
            "iat": now,
            "exp": now + 3600,
            "aud": "appstoreconnect-v1",
            "bid": self.bundle_id,
        }
        return jwt.encode(payload, self._private_key, algorithm="ES256", headers=headers)

    def _request(self, method: str, path: str) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers = {"Authorization": f"Bearer {self._generate_token()}"}
        resp = requests.request(method, url, headers=headers, timeout=20)
        if resp.status_code >= 400:
            raise AppleAppStoreServerAPIError(f"Apple API error {resp.status_code}: {resp.text}")
        return resp.json()

    def _get_keys(self) -> Dict[str, Any]:
        # Cache keys for 10 minutes
        if self._cached_keys and (time.time() - self._cached_keys_at) < 600:
            return self._cached_keys
        data = self._request("GET", "/inApps/v1/keys")
        self._cached_keys = data
        self._cached_keys_at = time.time()
        return data

    def _public_key_for_kid(self, kid: str):
        data = self._get_keys()
        for k in data.get("keys", []):
            if k.get("kid") == kid:
                jwk_json = jwt.api_jwk.PyJWK.from_dict(k).to_json()
                return jwt.algorithms.ECAlgorithm.from_jwk(jwk_json)
        raise AppleAppStoreServerAPIError(f"Apple JWS key not found for kid={kid}")

    def decode_and_verify_jws(self, jws: str) -> Dict[str, Any]:
        header = jwt.get_unverified_header(jws)
        kid = header.get("kid")
        if not kid:
            raise AppleAppStoreServerAPIError("Missing kid in JWS header")
        public_key = self._public_key_for_kid(kid)
        # Apple JWS payload doesn't use typical aud/iss for our needs; we validate signature + bundleId fields below.
        return jwt.decode(
            jws,
            key=public_key,
            algorithms=["ES256"],
            options={"verify_aud": False},
        )

    def get_transaction_info(self, transaction_id: str) -> AppleTransactionInfo:
        data = self._request("GET", f"/inApps/v1/transactions/{transaction_id}")
        signed = data.get("signedTransactionInfo")
        if not signed:
            raise AppleAppStoreServerAPIError("Apple response missing signedTransactionInfo")

        payload = self.decode_and_verify_jws(signed)

        # Basic sanity checks
        if payload.get("bundleId") and payload.get("bundleId") != self.bundle_id:
            raise AppleAppStoreServerAPIError("bundleId mismatch in signedTransactionInfo")

        product_id = payload.get("productId") or ""
        tx_id = payload.get("transactionId") or transaction_id
        original_tx_id = payload.get("originalTransactionId")
        expires_at = _utc_from_ms(payload.get("expiresDate"))

        return AppleTransactionInfo(
            product_id=product_id,
            transaction_id=tx_id,
            original_transaction_id=original_tx_id,
            expires_at=expires_at,
        )


