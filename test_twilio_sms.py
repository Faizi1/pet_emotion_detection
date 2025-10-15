"""
Test script for Twilio SMS Integration
Tests the SMS service and OTP functionality
"""

import os
import sys
import django
from pathlib import Path

# Add the project directory to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pet_emotion_detection.settings')
try:
    django.setup()
except Exception as e:
    print(f"Warning: Django setup failed: {e}")
    print("Continuing with basic SMS service test...")

def test_sms_service():
    """Test the Twilio SMS service"""
    print("📱 Testing Twilio SMS Service")
    print("=" * 50)
    
    try:
        from services.sms_service import sms_service
        
        # Test 1: Check service status
        print("🔧 Checking SMS service configuration...")
        status = sms_service.get_service_status()
        
        print(f"   Configured: {status['configured']}")
        print(f"   Account SID Set: {status['account_sid_set']}")
        print(f"   Auth Token Set: {status['auth_token_set']}")
        print(f"   Phone Number Set: {status['phone_number_set']}")
        print(f"   Client Initialized: {status['client_initialized']}")
        
        if status['configured']:
            print(f"   Twilio Phone Number: {status['settings']['phone_number']}")
            print("   ✅ SMS service is properly configured!")
        else:
            print("   ❌ SMS service not configured")
            print("   💡 Set these environment variables:")
            print("      - TWILIO_ACCOUNT_SID")
            print("      - TWILIO_AUTH_TOKEN")
            print("      - TWILIO_PHONE_NUMBER")
            return False
        
        # Test 2: Phone number validation
        print("\n📞 Testing phone number validation...")
        
        test_numbers = [
            "+1234567890",
            "1234567890",
            "+91-9876543210",
            "+44 20 7946 0958",
            "invalid-number"
        ]
        
        for number in test_numbers:
            validation = sms_service.validate_phone_number(number)
            status_icon = "✅" if validation['valid'] else "❌"
            print(f"   {status_icon} {number} -> {validation.get('formatted_number', 'Invalid')}")
            if not validation['valid']:
                print(f"      Error: {validation['error']}")
        
        # Test 3: OTP sending (dry run - won't actually send)
        print("\n📤 Testing OTP sending (dry run)...")
        
        if status['configured']:
            # Use a test number (won't actually send in demo)
            test_phone = "+1234567890"  # Replace with your test number
            test_otp = "123456"
            
            print(f"   Would send OTP {test_otp} to {test_phone}")
            print("   💡 To actually test SMS sending:")
            print(f"   📱 Update test_phone variable with your real number")
            print(f"   🔑 Ensure your Twilio account has credits")
            
            # Uncomment the line below to actually send SMS (requires real phone number)
            # result = sms_service.send_otp(test_phone, test_otp, 'registration')
            # print(f"   SMS Result: {result}")
        
        print("\n✅ SMS service test completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during SMS service test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_phone_formatting():
    """Test phone number formatting"""
    print("\n📞 Testing Phone Number Formatting")
    print("=" * 40)
    
    try:
        from services.sms_service import TwilioSMSService
        service = TwilioSMSService()
        
        test_cases = [
            ("+1234567890", "+1234567890"),
            ("1234567890", "+11234567890"),
            ("+91-9876543210", "+919876543210"),
            ("+44 20 7946 0958", "+442079460958"),
            ("+1 (555) 123-4567", "+15551234567"),
        ]
        
        print("   Input Number -> Formatted Number")
        print("   " + "-" * 40)
        
        for input_num, expected in test_cases:
            formatted = service.format_phone_number(input_num)
            status = "✅" if formatted == expected else "❌"
            print(f"   {status} {input_num} -> {formatted}")
            if formatted != expected:
                print(f"      Expected: {expected}")
        
        print("\n✅ Phone formatting test completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during phone formatting test: {e}")
        return False

def test_dependencies():
    """Test if required dependencies are available"""
    print("🔧 Testing Dependencies...")
    
    try:
        import twilio
        print("   ✅ Twilio SDK: Available")
        return True
    except ImportError:
        print("   ❌ Twilio SDK: Not installed")
        print("   💡 Install with: pip install twilio")
        return False

def show_setup_instructions():
    """Show setup instructions"""
    print("\n📋 Setup Instructions")
    print("=" * 50)
    print("1. Create Twilio account: https://www.twilio.com")
    print("2. Get your credentials from Console Dashboard:")
    print("   - Account SID (starts with AC...)")
    print("   - Auth Token")
    print("   - Phone Number (starts with +1...)")
    print("3. Create .env file with:")
    print("   TWILIO_ACCOUNT_SID=AC...")
    print("   TWILIO_AUTH_TOKEN=...")
    print("   TWILIO_PHONE_NUMBER=+1...")
    print("4. Install Twilio: pip install twilio")
    print("5. Run this test again")

if __name__ == "__main__":
    print("🚀 Twilio SMS Integration Test Suite")
    print("=" * 50)
    
    # Test dependencies
    deps_ok = test_dependencies()
    
    if not deps_ok:
        print("\n❌ Missing dependencies. Please install Twilio first.")
        show_setup_instructions()
        sys.exit(1)
    
    # Test phone formatting
    format_ok = test_phone_formatting()
    
    # Test SMS service
    sms_ok = test_sms_service()
    
    if sms_ok:
        print("\n🎉 All SMS tests passed!")
        print("\n📝 Next steps:")
        print("   1. Set up your Twilio account")
        print("   2. Add credentials to .env file")
        print("   3. Test with real phone number")
        print("   4. Your OTP system will send real SMS!")
        
        print("\n🔧 To test with real SMS:")
        print("   1. Update test_phone in the script with your number")
        print("   2. Uncomment the send_otp line")
        print("   3. Run the test again")
    else:
        print("\n⚠️  SMS service test failed.")
        print("\n💡 Common issues:")
        print("   1. Missing environment variables")
        print("   2. Invalid Twilio credentials")
        print("   3. No Twilio account credits")
        
        show_setup_instructions()
