# Backend IAP Requirements for iOS Subscriptions

This document outlines the requirements for implementing iOS In-App Purchase (IAP) receipt verification on the Python backend.

## Overview

The React Native app will send iOS receipt data to the backend for verification. The backend must:
1. Verify receipts with Apple's App Store Server API
2. Store subscription information in the database
3. Return subscription status to the app

## API Endpoints Required

### 1. Verify Receipt Endpoint

**Endpoint:** `POST /api/subscriptions/verify-receipt`

**Request Body:**
```json
{
  "receipt_data": "base64_encoded_receipt_string",
  "product_id": "com.petmood.premium.monthly",
  "transaction_id": "1000000123456789",
  "original_transaction_id": "1000000123456789" // Optional, for renewals
}
```

**Response (Success):**
```json
{
  "success": true,
  "subscription": {
    "plan_type": "premium", // or "family"
    "period": "monthly", // or "annual"
    "expires_at": "2024-02-17T10:00:00Z",
    "is_active": true
  }
}
```

**Response (Error):**
```json
{
  "success": false,
  "message": "Invalid receipt or subscription expired"
}
```

### 2. Get Subscription Status Endpoint

**Endpoint:** `GET /api/subscriptions/status`

**Headers:**
- `Authorization: Bearer <user_token>`

**Response:**
```json
{
  "subscription": {
    "plan_type": "premium",
    "period": "monthly",
    "expires_at": "2024-02-17T10:00:00Z",
    "is_active": true,
    "product_id": "com.petmood.premium.monthly"
  }
}
```

### 3. Restore Purchases Endpoint

**Endpoint:** `POST /api/subscriptions/restore`

**Headers:**
- `Authorization: Bearer <user_token>`

**Response:**
```json
{
  "subscriptions": [
    {
      "is_active": true,
      "plan_type": "premium",
      "period": "monthly",
      "expires_at": "2024-02-17T10:00:00Z",
      "product_id": "com.petmood.premium.monthly"
    }
  ]
}
```

## Product IDs

The following product IDs must be configured in App Store Connect:

1. **Premium Monthly:** `com.petmood.premium.monthly` - 7.99€
2. **Premium Annual:** `com.petmood.premium.annual` - 79.99€
3. **Family Monthly:** `com.petmood.family.monthly` - 9.99€
4. **Family Annual:** `com.petmood.family.annual` - 99.99€

## Backend Implementation Steps

### 1. Install Required Python Packages

```bash
pip install requests cryptography
```

### 2. Apple App Store Server API Setup

You'll need to:
- Create an App Store Connect API key
- Download the `.p8` key file
- Note the Key ID and Issuer ID

### 3. Receipt Verification Implementation

#### Option A: Using App Store Server API (Recommended for production)

```python
import requests
import jwt
import time
from datetime import datetime

class AppleReceiptVerifier:
    def __init__(self):
        self.key_id = "YOUR_KEY_ID"
        self.issuer_id = "YOUR_ISSUER_ID"
        self.bundle_id = "com.petmood"
        self.private_key_path = "path/to/AuthKey.p8"
        
    def generate_jwt_token(self):
        """Generate JWT token for App Store Server API"""
        with open(self.private_key_path, 'r') as f:
            private_key = f.read()
        
        headers = {
            "alg": "ES256",
            "kid": self.key_id,
            "typ": "JWT"
        }
        
        payload = {
            "iss": self.issuer_id,
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,
            "aud": "appstoreconnect-v1"
        }
        
        token = jwt.encode(payload, private_key, algorithm="ES256", headers=headers)
        return token
    
    def verify_receipt(self, receipt_data, product_id, transaction_id):
        """Verify receipt with App Store Server API"""
        token = self.generate_jwt_token()
        
        # Get transaction info
        url = f"https://api.storekit.itunes.apple.com/inApps/v1/transactions/{transaction_id}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            transaction_data = response.json()
            # Parse transaction data and verify subscription status
            return self.parse_transaction(transaction_data, product_id)
        else:
            return {"success": False, "message": "Receipt verification failed"}
    
    def parse_transaction(self, transaction_data, product_id):
        """Parse transaction data and extract subscription info"""
        # Map product IDs to plan types
        plan_mapping = {
            "com.petmood.premium.monthly": ("premium", "monthly"),
            "com.petmood.premium.annual": ("premium", "annual"),
            "com.petmood.family.monthly": ("family", "monthly"),
            "com.petmood.family.annual": ("family", "annual"),
        }
        
        plan_type, period = plan_mapping.get(product_id, (None, None))
        
        # Extract expiration date from transaction
        expires_at = transaction_data.get("signedRenewalInfo", {}).get("expiresDate")
        
        return {
            "success": True,
            "subscription": {
                "plan_type": plan_type,
                "period": period,
                "expires_at": expires_at,
                "is_active": self.is_subscription_active(expires_at)
            }
        }
    
    def is_subscription_active(self, expires_at):
        """Check if subscription is still active"""
        if not expires_at:
            return False
        expiry = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
        return expiry > datetime.now(expiry.tzinfo)
```

#### Option B: Using Legacy Receipt Validation API (For testing)

```python
import requests
import base64

def verify_receipt_legacy(receipt_data, is_sandbox=False):
    """Verify receipt using legacy App Store validation"""
    url = "https://buy.itunes.apple.com/verifyReceipt"
    if is_sandbox:
        url = "https://sandbox.itunes.apple.com/verifyReceipt"
    
    payload = {
        "receipt-data": receipt_data,
        "password": "YOUR_SHARED_SECRET"  # Optional, from App Store Connect
    }
    
    response = requests.post(url, json=payload)
    return response.json()
```

### 4. Database Schema

You'll need a subscription table:

```sql
CREATE TABLE subscriptions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    product_id VARCHAR(255) NOT NULL,
    plan_type VARCHAR(50) NOT NULL, -- 'premium' or 'family'
    period VARCHAR(50) NOT NULL, -- 'monthly' or 'annual'
    transaction_id VARCHAR(255) UNIQUE NOT NULL,
    original_transaction_id VARCHAR(255),
    expires_at TIMESTAMP NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_subscriptions_user_id ON subscriptions(user_id);
CREATE INDEX idx_subscriptions_transaction_id ON subscriptions(transaction_id);
CREATE INDEX idx_subscriptions_is_active ON subscriptions(is_active);
```

### 5. Example FastAPI/Django Implementation

#### FastAPI Example:

```python
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/api/subscriptions", tags=["subscriptions"])

class VerifyReceiptRequest(BaseModel):
    receipt_data: str
    product_id: str
    transaction_id: str
    original_transaction_id: Optional[str] = None

@router.post("/verify-receipt")
async def verify_receipt(
    request: VerifyReceiptRequest,
    current_user: User = Depends(get_current_user)
):
    verifier = AppleReceiptVerifier()
    result = verifier.verify_receipt(
        request.receipt_data,
        request.product_id,
        request.transaction_id
    )
    
    if result["success"]:
        # Save to database
        subscription = save_subscription(
            user_id=current_user.id,
            product_id=request.product_id,
            transaction_id=request.transaction_id,
            original_transaction_id=request.original_transaction_id,
            plan_type=result["subscription"]["plan_type"],
            period=result["subscription"]["period"],
            expires_at=result["subscription"]["expires_at"]
        )
    
    return result

@router.get("/status")
async def get_subscription_status(
    current_user: User = Depends(get_current_user)
):
    subscription = get_active_subscription(current_user.id)
    if not subscription:
        return {"subscription": None}
    
    return {
        "subscription": {
            "plan_type": subscription.plan_type,
            "period": subscription.period,
            "expires_at": subscription.expires_at.isoformat(),
            "is_active": subscription.is_active,
            "product_id": subscription.product_id
        }
    }
```

## Security Considerations

1. **Always verify receipts server-side** - Never trust client-side receipt data
2. **Use App Store Server API** - The legacy API is deprecated
3. **Store shared secret securely** - Use environment variables
4. **Handle subscription renewals** - Set up webhooks for subscription status updates
5. **Validate user authentication** - Ensure the user is authenticated before verifying receipts

## Testing

1. Use sandbox environment for testing
2. Create test accounts in App Store Connect
3. Test with sandbox receipts
4. Test subscription renewals and cancellations

## Webhooks (Optional but Recommended)

Set up App Store Server Notifications to receive real-time updates about subscription changes:
- Initial purchase
- Renewals
- Cancellations
- Refunds
- Expirations

Webhook endpoint: `POST /api/subscriptions/webhook`

## Additional Resources

- [App Store Server API Documentation](https://developer.apple.com/documentation/appstoreserverapi)
- [Receipt Validation Guide](https://developer.apple.com/documentation/appstorereceipts)
- [App Store Connect API](https://developer.apple.com/documentation/appstoreconnectapi)
