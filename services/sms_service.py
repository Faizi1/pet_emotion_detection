"""
SMS Service for OTP and notifications
Currently using: Vonage (Twilio and Telgorithm are available but commented out)
Set USE_VONAGE=true in .env to enable Vonage SMS
"""

import os
import logging
from typing import Dict, Optional, Tuple
from django.conf import settings

logger = logging.getLogger(__name__)

class TwilioSMSService:
    """
    Twilio SMS service for sending OTP and notifications
    """
    
    def __init__(self):
        self.account_sid = settings.TWILIO_ACCOUNT_SID
        self.auth_token = settings.TWILIO_AUTH_TOKEN
        self.phone_number = settings.TWILIO_PHONE_NUMBER
        self.client = None
        self.is_configured = bool(self.account_sid and self.auth_token and self.phone_number)
        
        if self.is_configured:
            try:
                from twilio.rest import Client
                self.client = Client(self.account_sid, self.auth_token)
                logger.info("Twilio SMS service initialized successfully")
            except ImportError:
                logger.error("Twilio SDK not installed. Run: pip install twilio")
                self.is_configured = False
            except Exception as e:
                logger.error(f"Twilio initialization failed: {e}")
                self.is_configured = False
        else:
            logger.warning("Twilio not configured. Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER")
    
    def format_phone_number(self, phone_number: str) -> str:
        """
        Format phone number to E.164 format
        Examples:
        - +1234567890 -> +1234567890
        - 1234567890 -> +1234567890
        - +91-9876543210 -> +919876543210
        """
        # Remove all non-digit characters except +
        cleaned = ''.join(char for char in phone_number if char.isdigit() or char == '+')
        
        # Add + if not present
        if not cleaned.startswith('+'):
            # Assume US number if no country code
            if len(cleaned) == 10:
                cleaned = '+1' + cleaned
            else:
                cleaned = '+' + cleaned
        print('number formate', cleaned)
        return cleaned
    
    def send_otp(self, phone_number: str, otp_code: str, message_type: str = "registration") -> Dict[str, any]:
        """
        Send OTP code via SMS
        
        Args:
            phone_number: Recipient phone number
            otp_code: 6-digit OTP code
            message_type: Type of OTP (registration, login, reset, etc.)
            
        Returns:
            Dict with success status and details
        """
        if not self.is_configured:
            return {
                'success': False,
                'error': 'Twilio not configured',
                'message': 'SMS service not available'
            }
        
        try:
            # Format phone number
            formatted_number = self.format_phone_number(phone_number)
            
            # Create message based on type
            messages = {
                'registration': f"Your Pet Mood verification code is: {otp_code}. Valid for 10 minutes.",
                'login': f"Your Pet Mood login code is: {otp_code}. Valid for 10 minutes.",
                'reset': f"Your Pet Mood password reset code is: {otp_code}. Valid for 10 minutes.",
                'general': f"Your Pet Mood verification code is: {otp_code}. Valid for 10 minutes."
            }
            
            message_body = messages.get(message_type, messages['general'])
            
            # Send SMS
            message = self.client.messages.create(
                body=message_body,
                from_=self.phone_number,
                to=formatted_number
            )
            
            logger.info(f"SMS sent successfully. SID: {message.sid}, To: {formatted_number}")
            
            return {
                'success': True,
                'message_sid': message.sid,
                'to': formatted_number,
                'status': message.status,
                'error': None
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"SMS sending failed: {error_msg}")
            
            return {
                'success': False,
                'error': error_msg,
                'message': 'Failed to send SMS'
            }
    
    def send_notification(self, phone_number: str, message: str) -> Dict[str, any]:
        """
        Send custom notification message
        
        Args:
            phone_number: Recipient phone number
            message: Custom message to send
            
        Returns:
            Dict with success status and details
        """
        if not self.is_configured:
            return {
                'success': False,
                'error': 'Twilio not configured',
                'message': 'SMS service not available'
            }
        
        try:
            # Format phone number
            formatted_number = self.format_phone_number(phone_number)
            
            # Send SMS
            twilio_message = self.client.messages.create(
                body=message,
                from_=self.phone_number,
                to=formatted_number
            )
            
            logger.info(f"Notification sent successfully. SID: {twilio_message.sid}, To: {formatted_number}")
            
            return {
                'success': True,
                'message_sid': twilio_message.sid,
                'to': formatted_number,
                'status': twilio_message.status,
                'error': None
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Notification sending failed: {error_msg}")
            
            return {
                'success': False,
                'error': error_msg,
                'message': 'Failed to send notification'
            }
    
    def get_message_status(self, message_sid: str) -> Dict[str, any]:
        """
        Get status of a sent message
        
        Args:
            message_sid: Twilio message SID
            
        Returns:
            Dict with message status and details
        """
        if not self.is_configured:
            return {
                'success': False,
                'error': 'Twilio not configured'
            }
        
        try:
            message = self.client.messages(message_sid).fetch()
            
            # Enhanced response with delivery status interpretation
            status_info = {
                'success': True,
                'message_sid': message.sid,
                'status': message.status,
                'to': message.to,
                'from': message.from_,
                'body': message.body,
                'date_created': str(message.date_created) if message.date_created else None,
                'date_sent': str(message.date_sent) if message.date_sent else None,
                'error_code': message.error_code,
                'error_message': message.error_message
            }
            
            # Add status interpretation
            if message.status == 'delivered':
                status_info['delivery_status'] = 'success'
                status_info['delivery_message'] = 'SMS delivered successfully'
            elif message.status == 'undelivered':
                status_info['delivery_status'] = 'failed'
                status_info['delivery_message'] = 'SMS could not be delivered'
                if message.error_code == 30034:
                    status_info['delivery_message'] += ' - Invalid destination address'
                elif message.error_code == 21211:
                    status_info['delivery_message'] += ' - Invalid phone number format'
                elif message.error_code == 21614:
                    status_info['delivery_message'] += ' - Phone number not reachable'
            elif message.status == 'failed':
                status_info['delivery_status'] = 'failed'
                status_info['delivery_message'] = 'SMS sending failed'
            else:
                status_info['delivery_status'] = 'pending'
                status_info['delivery_message'] = f'SMS status: {message.status}'
            
            return status_info
            
        except Exception as e:
            logger.error(f"Failed to fetch message status: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def validate_phone_number(self, phone_number: str) -> Dict[str, any]:
        """
        Validate phone number format and Twilio compatibility
        
        Args:
            phone_number: Phone number to validate
            
        Returns:
            Dict with validation results
        """
        try:
            formatted_number = self.format_phone_number(phone_number)
            
            # Basic validation
            if not formatted_number.startswith('+'):
                return {
                    'valid': False,
                    'error': 'Phone number must include country code',
                    'suggestion': 'Add country code (e.g., +1 for US, +91 for India)'
                }
            
            # Check length (minimum 10 digits)
            digits_only = ''.join(char for char in formatted_number if char.isdigit())
            if len(digits_only) < 10:
                return {
                    'valid': False,
                    'error': 'Phone number too short',
                    'suggestion': 'Ensure phone number has at least 10 digits'
                }
            
            return {
                'valid': True,
                'formatted_number': formatted_number,
                'error': None
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': f'Validation failed: {str(e)}'
            }
    
    def get_service_status(self) -> Dict[str, any]:
        """
        Get Twilio service configuration status
        
        Returns:
            Dict with service status and configuration
        """
        return {
            'configured': self.is_configured,
            'account_sid_set': bool(self.account_sid),
            'auth_token_set': bool(self.auth_token),
            'phone_number_set': bool(self.phone_number),
            'client_initialized': bool(self.client),
            'settings': {
                'account_sid': self.account_sid[:8] + '...' if self.account_sid else None,
                'phone_number': self.phone_number,
                'auth_token_set': bool(self.auth_token)
            }
        }


# Telgorithm SMS Service Implementation
class TelgorithmSMSService:
    """
    Telgorithm SMS service for sending OTP and notifications
    Alternative to Twilio with better pricing and global coverage
    """
    
    def __init__(self):
        from django.conf import settings
        self.api_key = getattr(settings, 'TELGORITHM_API_KEY', None)
        self.sender_id = getattr(settings, 'TELGORITHM_SENDER_ID', 'PetMood')
        self.api_url = getattr(settings, 'TELGORITHM_API_URL', 'https://api.telgorithm.com/v1/sms')
        self.is_configured = bool(self.api_key)
        
        if self.is_configured:
            logger.info("Telgorithm SMS service initialized successfully")
        else:
            logger.warning("Telgorithm not configured. Set TELGORITHM_API_KEY")
    
    def format_phone_number(self, phone_number: str) -> str:
        """Format phone number to E.164 format (same as Twilio)"""
        # Remove all non-digit characters except +
        cleaned = ''.join(char for char in phone_number if char.isdigit() or char == '+')
        
        # Add + if not present
        if not cleaned.startswith('+'):
            if len(cleaned) == 10:
                cleaned = '+1' + cleaned  # Assume US
            else:
                cleaned = '+' + cleaned
        
        return cleaned
    
    def send_otp(self, phone_number: str, otp_code: str, message_type: str = "registration") -> Dict[str, any]:
        """Send OTP code via Telgorithm SMS"""
        if not self.is_configured:
            return {
                'success': False,
                'error': 'Telgorithm not configured',
                'message': 'SMS service not available'
            }
        
        try:
            
            import requests
            
            # Format phone number
            formatted_number = self.format_phone_number(phone_number)
            
            # Create message
            messages = {
                'registration': f"Your Pet Mood verification code is: {otp_code}. Valid for 10 minutes.",
                'login': f"Your Pet Mood login code is: {otp_code}. Valid for 10 minutes.",
                'reset': f"Your Pet Mood password reset code is: {otp_code}. Valid for 10 minutes.",
                'general': f"Your Pet Mood verification code is: {otp_code}. Valid for 10 minutes."
            }
            
            message_body = messages.get(message_type, messages['general'])
            
            # Send SMS via Telgorithm API
            response = requests.post(
                f"{self.api_url}/send",
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'to': formatted_number,
                    'from': self.sender_id,
                    'message': message_body
                },
                timeout=10
            )
            
            if response.status_code == 200 or response.status_code == 201:
                data = response.json()
                logger.info(f"Telgorithm SMS sent successfully to: {formatted_number}")
                
                return {
                    'success': True,
                    'message_id': data.get('message_id', data.get('id', 'unknown')),
                    'to': formatted_number,
                    'status': 'sent',
                    'error': None,
                    'provider': 'telgorithm'
                }
            else:
                error_msg = f"Telgorithm API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return {
                    'success': False,
                    'error': error_msg,
                    'message': 'Failed to send SMS via Telgorithm'
                }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Telgorithm SMS sending failed: {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'message': 'Failed to send SMS via Telgorithm'
            }
    
    def send_notification(self, phone_number: str, message: str) -> Dict[str, any]:
        """Send custom notification message via Telgorithm"""
        return self.send_otp(phone_number, message, 'general')
    
    def get_service_status(self) -> Dict[str, any]:
        """Get Telgorithm service configuration status"""
        return {
            'configured': self.is_configured,
            'api_key_set': bool(self.api_key),
            'sender_id': self.sender_id,
            'api_url': self.api_url
        }


# Hybrid SMS Service - Uses Telgorithm with Twilio fallback
class HybridSMSService:
    """
    Hybrid SMS service that tries Telgorithm first, falls back to Twilio
    """
    
    def __init__(self):
        from django.conf import settings
        self.use_telgorithm = getattr(settings, 'USE_TELGORITHM', False)
        
        # Initialize both services
        self.telgorithm = TelgorithmSMSService()
        self.twilio = TwilioSMSService()
        
        # Determine primary service
        self.primary_service = self.telgorithm if self.use_telgorithm else self.twilio
        self.fallback_service = self.twilio if self.use_telgorithm else None
        
        logger.info(f"Hybrid SMS service initialized - Primary: {'Telgorithm' if self.use_telgorithm else 'Twilio'}")
    
    def send_otp(self, phone_number: str, otp_code: str, message_type: str = "registration") -> Dict[str, any]:
        """Send OTP via primary service, fallback to secondary if fails"""
        # Try primary service
        result = self.primary_service.send_otp(phone_number, otp_code, message_type)
        
        # If primary fails and we have fallback, try fallback
        if not result['success'] and self.fallback_service and self.fallback_service.is_configured:
            logger.info(f"Primary SMS service failed, trying fallback: {self.fallback_service.__class__.__name__}")
            fallback_result = self.fallback_service.send_otp(phone_number, otp_code, message_type)
            fallback_result['fallback_used'] = True
            fallback_result['primary_failed'] = result.get('error', 'Unknown error')
            return fallback_result
        
        return result
    
    def send_notification(self, phone_number: str, message: str) -> Dict[str, any]:
        """Send notification via primary service, fallback to secondary if fails"""
        result = self.primary_service.send_notification(phone_number, message)
        
        if not result['success'] and self.fallback_service and self.fallback_service.is_configured:
            logger.info(f"Primary SMS service failed, trying fallback: {self.fallback_service.__class__.__name__}")
            fallback_result = self.fallback_service.send_notification(phone_number, message)
            fallback_result['fallback_used'] = True
            return fallback_result
        
        return result
    
    def get_service_status(self) -> Dict[str, any]:
        """Get status of both services"""
        return {
            'primary_service': 'Telgorithm' if self.use_telgorithm else 'Twilio',
            'fallback_service': 'Twilio' if self.use_telgorithm else None,
            'telgorithm_status': self.telgorithm.get_service_status(),
            'twilio_status': self.twilio.get_service_status()
        }


# Vonage SMS Service Implementation
class VonageSMSService:
    """
    Vonage (formerly Nexmo) SMS service for sending OTP and notifications
    Excellent global coverage with better pricing than Twilio
    """
    
    def __init__(self):
        from django.conf import settings
        self.api_key = getattr(settings, 'VONAGE_API_KEY', None)
        self.api_secret = getattr(settings, 'VONAGE_API_SECRET', None)
        # Default to your Vonage phone number if set, otherwise use custom sender
        # Try phone number first: +13863783649 (already registered in Vonage)
        self.sender_id = getattr(settings, 'VONAGE_SENDER_ID', '+13863783649')
        self.api_url = 'https://rest.nexmo.com/sms/json'
        self.is_configured = bool(self.api_key and self.api_secret)
        
        if self.is_configured:
            logger.info("Vonage SMS service initialized successfully")
        else:
            logger.warning("Vonage not configured. Set VONAGE_API_KEY and VONAGE_API_SECRET")
    
    def format_phone_number(self, phone_number: str) -> str:
        """Format phone number to E.164 format (same as Twilio)"""
        # Remove all non-digit characters except +
        cleaned = ''.join(char for char in phone_number if char.isdigit() or char == '+')
        
        # Add + if not present
        if not cleaned.startswith('+'):
            if len(cleaned) == 10:
                cleaned = '+1' + cleaned  # Assume US
            else:
                cleaned = '+' + cleaned
        
        return cleaned
    
    def send_otp(self, phone_number: str, otp_code: str, message_type: str = "registration") -> Dict[str, any]:
        """Send OTP code via Vonage SMS"""
        if not self.is_configured:
            return {
                'success': False,
                'error': 'Vonage not configured',
                'message': 'SMS service not available'
            }
        
        try:
            import requests
            
            # Format phone number
            formatted_number = self.format_phone_number(phone_number)
            
            # Create message
            messages = {
                'registration': f"Your Pet Mood verification code is: {otp_code}. Valid for 10 minutes.",
                'login': f"Your Pet Mood login code is: {otp_code}. Valid for 10 minutes.",
                'reset': f"Your Pet Mood password reset code is: {otp_code}. Valid for 10 minutes.",
                'general': f"Your Pet Mood verification code is: {otp_code}. Valid for 10 minutes."
            }
            
            message_body = messages.get(message_type, messages['general'])
            
            # Send SMS via Vonage API
            payload = {
                'api_key': self.api_key,
                'api_secret': self.api_secret,
                'to': formatted_number,
                'from': self.sender_id,
                'text': message_body,
                'type': 'text'
            }
            
            response = requests.post(
                self.api_url,
                data=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Check if message was sent successfully
                if data.get('messages') and len(data['messages']) > 0:
                    message_info = data['messages'][0]
                    status = message_info.get('status', 'unknown')
                    
                    if status == '0':  # Success
                        logger.info(f"Vonage SMS sent successfully to: {formatted_number}")
                        return {
                            'success': True,
                            'message_id': message_info.get('message-id', 'unknown'),
                            'to': formatted_number,
                            'status': 'sent',
                            'error': None,
                            'provider': 'vonage'
                        }
                    else:
                        error_msg = f"Vonage API error: {message_info.get('error-text', 'Unknown error')}"
                        logger.error(error_msg)
                        return {
                            'success': False,
                            'error': error_msg,
                            'message': 'Failed to send SMS via Vonage'
                        }
                else:
                    error_msg = f"Vonage API error: {data.get('error-text', 'Unknown error')}"
                    logger.error(error_msg)
                    return {
                        'success': False,
                        'error': error_msg,
                        'message': 'Failed to send SMS via Vonage'
                    }
            else:
                error_msg = f"Vonage API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return {
                    'success': False,
                    'error': error_msg,
                    'message': 'Failed to send SMS via Vonage'
                }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Vonage SMS sending failed: {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'message': 'Failed to send SMS via Vonage'
            }
    
    def send_notification(self, phone_number: str, message: str) -> Dict[str, any]:
        """Send custom notification message via Vonage"""
        return self.send_otp(phone_number, message, 'general')
    
    def get_service_status(self) -> Dict[str, any]:
        """Get Vonage service configuration status"""
        return {
            'configured': self.is_configured,
            'api_key_set': bool(self.api_key),
            'api_secret_set': bool(self.api_secret),
            'sender_id': self.sender_id
        }


# Enhanced Hybrid SMS Service - Now supports Vonage, Telgorithm, and Twilio
class EnhancedHybridSMSService:
    """
    Enhanced hybrid SMS service that supports Vonage, Telgorithm, and Twilio
    Tries services in priority order with automatic fallback
    """
    
    def __init__(self):
        from django.conf import settings
        self.use_vonage = getattr(settings, 'USE_VONAGE', False)
        self.use_telgorithm = getattr(settings, 'USE_TELGORITHM', False)
        
        # Initialize all services
        self.vonage = VonageSMSService()
        self.telgorithm = TelgorithmSMSService()
        self.twilio = TwilioSMSService()
        
        # Determine priority order
        if self.use_vonage and self.vonage.is_configured:
            self.primary_service = self.vonage
            self.fallback_services = [self.telgorithm, self.twilio]
            logger.info("Enhanced Hybrid SMS service - Primary: Vonage")
        elif self.use_telgorithm and self.telgorithm.is_configured:
            self.primary_service = self.telgorithm
            self.fallback_services = [self.vonage, self.twilio]
            logger.info("Enhanced Hybrid SMS service - Primary: Telgorithm")
        elif self.twilio.is_configured:
            self.primary_service = self.twilio
            self.fallback_services = [self.vonage, self.telgorithm]
            logger.info("Enhanced Hybrid SMS service - Primary: Twilio")
        else:
            # No service configured
            self.primary_service = None
            self.fallback_services = []
            logger.warning("No SMS service configured!")
    
    def send_otp(self, phone_number: str, otp_code: str, message_type: str = "registration") -> Dict[str, any]:
        """Send OTP via primary service, try fallbacks if fails"""
        if not self.primary_service:
            return {
                'success': False,
                'error': 'No SMS service configured',
                'message': 'Please configure VONAGE, TELGORITHM, or TWILIO credentials'
            }
        
        # Try primary service
        result = self.primary_service.send_otp(phone_number, otp_code, message_type)
        
        # If primary succeeds, return result
        if result['success']:
            return result
        
        # Try fallback services
        if self.fallback_services:
            for fallback in self.fallback_services:
                if fallback and fallback.is_configured:
                    logger.info(f"Primary SMS failed, trying fallback: {fallback.__class__.__name__}")
                    fallback_result = fallback.send_otp(phone_number, otp_code, message_type)
                    
                    if fallback_result['success']:
                        fallback_result['fallback_used'] = True
                        fallback_result['primary_failed'] = result.get('error', 'Unknown error')
                        return fallback_result
        
        # All services failed
        return result
    
    def send_notification(self, phone_number: str, message: str) -> Dict[str, any]:
        """Send notification via primary service, try fallbacks if fails"""
        if not self.primary_service:
            return {
                'success': False,
                'error': 'No SMS service configured'
            }
        
        result = self.primary_service.send_notification(phone_number, message)
        
        if result['success']:
            return result
        
        # Try fallback services
        if self.fallback_services:
            for fallback in self.fallback_services:
                if fallback and fallback.is_configured:
                    fallback_result = fallback.send_notification(phone_number, message)
                    if fallback_result['success']:
                        fallback_result['fallback_used'] = True
                        return fallback_result
        
        return result
    
    def get_service_status(self) -> Dict[str, any]:
        """Get status of all services"""
        return {
            'primary_service': self.primary_service.__class__.__name__ if self.primary_service else 'None',
            'fallback_services': [svc.__class__.__name__ for svc in self.fallback_services if svc and svc.is_configured],
            'vonage_status': self.vonage.get_service_status(),
            'telgorithm_status': self.telgorithm.get_service_status(),
            'twilio_status': self.twilio.get_service_status()
        }


# Global instance - Vonage Only (Telgorithm and Twilio commented out)
# Set USE_VONAGE=true to use Vonage as primary

# Option 1: Use Vonage only (uncomment to use)
from django.conf import settings
USE_VONAGE = getattr(settings, 'USE_VONAGE', False)

if USE_VONAGE:
    sms_service = VonageSMSService()
else:
    # Fallback to Twilio if Vonage not configured
    # sms_service = TwilioSMSService()  # Commented out
    # sms_service = TelgorithmSMSService()  # Commented out
    # sms_service = VonageSMSService()  # Commented out
    # Use simple Vonage-only service
    sms_service = VonageSMSService()
