"""
Test different Vonage sender IDs to find which works
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Get Vonage credentials
api_key = os.getenv('VONAGE_API_KEY', '')
api_secret = os.getenv('VONAGE_API_SECRET', '')
test_number = input("Enter phone number to test: ")

print("=" * 60)
print("Testing Different Vonage Sender IDs")
print("=" * 60)

# Test different sender formats
senders = [
    ('+13863783649', 'Your Vonage Number (with +)'),
    ('13863783649', 'Your Vonage Number (without +)'),
    ('Vonage', 'Vonage Default'),
    ('PetMood', 'Custom Text'),
    ('', 'Empty (Use account default)'),
    ('VONAGE', 'Vonage Uppercase'),
]

for sender, description in senders:
    print(f"\nTesting: {description}")
    print(f"  Sender: '{sender}'")
    
    try:
        payload = {
            'api_key': api_key,
            'api_secret': api_secret,
            'to': test_number,
            'from': sender,
            'text': f'Test from {sender}' if sender else 'Test message',
            'type': 'text'
        }
        
        response = requests.post(
            'https://rest.nexmo.com/sms/json',
            data=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"  Status: {response.status_code}")
            
            if data.get('messages'):
                msg = data['messages'][0]
                status = msg.get('status', 'unknown')
                
                if status == '0':
                    print(f"  ✅ SUCCESS! Message ID: {msg.get('message-id')}")
                    print(f"\n🎉 WORKING SENDER: '{sender}'")
                    break
                else:
                    print(f"  ❌ Failed: {msg.get('error-text', 'Unknown error')}")
            else:
                print(f"  ❌ Error: {data.get('error-text', 'Unknown error')}")
        else:
            print(f"  ❌ HTTP Error: {response.status_code}")
            
    except Exception as e:
        print(f"  ❌ Exception: {e}")

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)
