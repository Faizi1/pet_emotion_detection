"""
Twilio SMS Service for OTP and notifications
Handles SMS sending through Twilio API
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


# Global instance
sms_service = TwilioSMSService()
