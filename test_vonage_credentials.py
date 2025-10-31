"""
Test Vonage SMS Credentials
Run this to debug "Bad Credentials" error
"""

import os
import sys
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_vonage_credentials():
    """Test if Vonage credentials are valid"""
    
    print("=" * 60)
    print("Testing Vonage SMS Credentials")
    print("=" * 60)
    
    # Get credentials
    api_key = os.getenv('VONAGE_API_KEY', '')
    api_secret = os.getenv('VONAGE_API_SECRET', '')
    use_vonage = os.getenv('USE_VONAGE', 'false').lower()
    
    print(f"\n1. Configuration Check:")
    print(f"   USE_VONAGE: {use_vonage}")
    print(f"   VONAGE_API_KEY: {'✓ SET' if api_key else '✗ NOT SET'}")
    print(f"   VONAGE_API_SECRET: {'✓ SET' if api_secret else '✗ NOT SET'}")
    
    if not api_key:
        print("\n❌ ERROR: VONAGE_API_KEY is not set in .env file!")
        print("   Add: VONAGE_API_KEY=your_key_here")
        return False
    
    if not api_secret:
        print("\n❌ ERROR: VONAGE_API_SECRET is not set in .env file!")
        print("   Add: VONAGE_API_SECRET=your_secret_here")
        return False
    
    print(f"\n   API Key Length: {len(api_key)}")
    print(f"   API Secret Length: {len(api_secret)}")
    
    # Check format
    print(f"\n2. Format Check:")
    if len(api_key) < 8:
        print("   ⚠️  API Key seems too short (should be 8+ characters)")
    if len(api_secret) < 8:
        print("   ⚠️  API Secret seems too short (should be 8+ characters)")
    
    # Test credentials with Vonage API
    print(f"\n3. Testing Credentials with Vonage API...")
    
    try:
        # Test 1: Check account balance
        print("   Testing: Account Balance Check")
        response = requests.get(
            'https://rest.nexmo.com/account/get-balance',
            params={
                'api_key': api_key,
                'api_secret': api_secret
            },
            timeout=10
        )
        
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            balance = data.get('value', 0)
            print(f"\n✅ SUCCESS: Credentials are VALID!")
            print(f"   Account Balance: ${balance}")
            print(f"   Auto Reload: {data.get('autoReload', False)}")
            return True
        elif response.status_code == 401:
            print("\n❌ FAILED: Bad Credentials!")
            print("   Your API key or secret is incorrect.")
            print("   Check: https://dashboard.nexmo.com/ for correct credentials")
            return False
        else:
            print(f"\n❌ FAILED: Status {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("\n❌ FAILED: Request timed out")
        print("   Check your internet connection")
        return False
    except requests.exceptions.RequestException as e:
        print(f"\n❌ FAILED: Request error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        return False

def test_send_sms():
    """Test sending SMS with Vonage"""
    
    api_key = os.getenv('VONAGE_API_KEY', '')
    api_secret = os.getenv('VONAGE_API_SECRET', '')
    
    if not api_key or not api_secret:
        print("\n⚠️  Cannot test SMS - credentials not set")
        return
    
    print("\n" + "=" * 60)
    print("Testing SMS Sending")
    print("=" * 60)
    
    # Get test phone number
    test_number = input("\nEnter test phone number (or 'skip' to skip): ")
    
    if test_number.lower() == 'skip':
        print("   Skipping SMS test")
        return
    
    if not test_number:
        print("   Skipping SMS test")
        return
    
    payload = {
        'api_key': api_key,
        'api_secret': api_secret,
        'to': test_number,
        'from': 'PetMood',
        'text': 'Test SMS from Pet Mood OTP service'
    }
    
    print(f"\n   Sending SMS to: {test_number}")
    
    try:
        response = requests.post(
            'https://rest.nexmo.com/sms/json',
            data=payload,
            timeout=10
        )
        
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('messages'):
                msg = data['messages'][0]
                if msg.get('status') == '0':
                    print("\n✅ SMS sent successfully!")
                    print(f"   Message ID: {msg.get('message-id')}")
                else:
                    print(f"\n❌ SMS failed: {msg.get('error-text')}")
            else:
                print(f"\n❌ Error: {data.get('error-text', 'Unknown error')}")
        
    except Exception as e:
        print(f"\n❌ Failed to send SMS: {e}")

if __name__ == '__main__':
    # Test credentials
    success = test_vonage_credentials()
    
    # If credentials are valid, test SMS
    if success:
        test_send_sms()
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
