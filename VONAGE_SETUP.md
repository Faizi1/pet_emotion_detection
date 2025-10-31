# Vonage SMS Service - Setup & Usage

## 🎯 What is Vonage?

**Vonage** is a leading SMS service provider that offers:
- ✅ Global coverage (200+ countries)
- ✅ Lower costs than Twilio ($0.0055/SMS)
- ✅ 99%+ delivery rate
- ✅ Simple REST API
- ✅ Free $2 trial credits

---

## 📋 Quick Setup

### Step 1: Sign Up
1. Go to https://dashboard.nexmo.com/
2. Sign up (free account)
3. Get $2 free credits

### Step 2: Get Credentials
1. Login to dashboard
2. Find **API Key** and **API Secret**
3. Copy them exactly

### Step 3: Configure
Add to `.env` file:
```env
USE_VONAGE=true
VONAGE_API_KEY=your_api_key_here
VONAGE_API_SECRET=your_api_secret_here
VONAGE_SENDER_ID=13863783649  # Your Vonage phone number without +
```

### Step 4: Test
```bash
python test_vonage_credentials.py
```

Should show: "✅ SUCCESS: Credentials are VALID!"

---

## 🧪 Testing

### Test Credentials
```bash
python test_vonage_credentials.py
```

### Test Sender IDs
```bash
python test_vonage_sender.py
```

### Check Configuration
```bash
python check_vonage_setup.py
```

---

## ⚙️ Configuration

### Sender ID Format

**Use your Vonage phone number:**
```env
VONAGE_SENDER_ID=13863783649  # WITHOUT + sign!
```

**Don't use:**
- ❌ `VONAGE_SENDER_ID=+13863783649` (with +)
- ❌ `VONAGE_SENDER_ID=PetMood` (needs registration)

**Use:**
- ✅ `VONAGE_SENDER_ID=13863783649` (digits only)

---

## ⚠️ Common Issues

### Issue 1: "Bad Credentials"
**Solution:** Get fresh credentials from https://dashboard.nexmo.com/
- Copy API Key and Secret EXACTLY
- No extra spaces
- No quotes

### Issue 2: "Illegal Sender Address"
**Solution:** Use your Vonage phone number without + sign
```env
VONAGE_SENDER_ID=13863783649  # Correct!
```

### Issue 3: "[FREE SMS DEMO, TEST MESSAGE]"
**What it means:** Working correctly! Just on trial account.
**To remove:** Upgrade to paid Vonage account (add payment method)

---

## 💰 Pricing

**Vonage SMS Costs:**
- US: $0.0055/SMS
- UK: $0.0065/SMS
- India: $0.003/SMS
- Global average: $0.01/SMS

**Monthly Example (1000 SMS):**
- US: $5.50
- Global: $3-10

**Savings vs Twilio:** 60-80% cheaper!

---

## 🚀 API Usage

### In Django Views
```python
from services.sms_service import sms_service

result = sms_service.send_otp(
    phone_number="+1234567890",
    otp_code="123456",
    message_type="registration"
)

print(result)
# Should show: {'success': True, 'provider': 'vonage'}
```

---

## 📊 What's Working

✅ **SMS Delivery:** Working perfectly  
✅ **OTP Codes:** Delivered instantly  
✅ **Credential Validation:** Tested  
✅ **Sender ID:** Configured correctly  
✅ **Cost:** 60-80% cheaper than Twilio  

---

## 📝 Files in This Project

**Main Code:**
- `services/sms_service.py` - Vonage SMS service

**Test Files:**
- `test_vonage_credentials.py` - Test credentials
- `test_vonage_sender.py` - Test sender IDs
- `check_vonage_setup.py` - Check configuration

**Documentation:**
- `VONAGE_SETUP.md` - This file

---

## ✅ Quick Start

1. Add credentials to `.env`
2. Set `USE_VONAGE=true`
3. Run test: `python test_vonage_credentials.py`
4. Restart Django: `python manage.py runserver`
5. Done! SMS now uses Vonage

---

**Need help?** Check the test files or Vonage dashboard!
