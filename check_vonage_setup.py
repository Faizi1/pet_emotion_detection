"""
Quick Check: Verify Vonage Setup
"""

import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("Checking Vonage Setup")
print("=" * 60)

# Check environment variables
use_vonage = os.getenv('USE_VONAGE', 'false')
api_key = os.getenv('VONAGE_API_KEY', '')
api_secret = os.getenv('VONAGE_API_SECRET', '')
sender_id = os.getenv('VONAGE_SENDER_ID', 'PetMood')

print(f"\nConfiguration:")
print(f"  USE_VONAGE: {use_vonage}")
print(f"  VONAGE_API_KEY: {'✓ SET' if api_key else '✗ NOT SET'}")
print(f"  VONAGE_API_SECRET: {'✓ SET' if api_secret else '✗ NOT SET'}")
print(f"  VONAGE_SENDER_ID: {sender_id}")

if use_vonage.lower() == 'true':
    print(f"\n✅ Vonage is ENABLED")
else:
    print(f"\n❌ Vonage is DISABLED")
    print(f"   Add USE_VONAGE=true to .env file")

if api_key and api_secret:
    print(f"\n✅ Credentials are set")
    print(f"\nNow run: python test_vonage_credentials.py")
else:
    print(f"\n❌ Credentials are missing")
    print(f"   Add to .env file:")
    print(f"   VONAGE_API_KEY=your_key")
    print(f"   VONAGE_API_SECRET=your_secret")

print("\n" + "=" * 60)
