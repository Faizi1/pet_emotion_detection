"""
Nyckel Dog Emotions Identifier Service for Pet Emotion Detection (Images)

This service:
- Uses Nyckel's pretrained "dog-emotions-identifier" classifier
- Returns emotions: Happy, Sad, Angry, Relaxed (maps to our pet emotions)
- Returns a unified dict compatible with the existing scan response

Authentication (either one):
- NYCKEL_BEARER_TOKEN - direct Bearer token
- NYCKEL_CLIENT_ID + NYCKEL_CLIENT_SECRET - OAuth client credentials (token fetched from connect/token)

Reference: https://www.nyckel.com/pretrained-classifiers/dog-emotions-identifier/
Docs: https://www.nyckel.com/docs (Authentication / Create access token)
"""

import logging
import os
import base64
import time
import requests
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Token endpoint per Nyckel docs
NYCKEL_TOKEN_URL = "https://www.nyckel.com/connect/token"


class NyckelService:
    """Wrapper around Nyckel API for pet emotion detection."""

    def __init__(self) -> None:
        self.bearer_token = os.getenv("NYCKEL_BEARER_TOKEN", "").strip()
        self.client_id = os.getenv("NYCKEL_CLIENT_ID", "").strip()
        self.client_secret = os.getenv("NYCKEL_CLIENT_SECRET", "").strip()
        self.function_name = "dog-emotions-identifier"
        self.api_url = f"https://www.nyckel.com/v1/functions/{self.function_name}/invoke"
        
        # Cache for token from client credentials (expires in 1 hour)
        self._token_expires_at = 0.0
        
        self._initialized = False
        
        if self.bearer_token:
            self._initialized = True
            logger.info("Nyckel service initialized with Bearer token")
        elif self.client_id and self.client_secret:
            try:
                token = self._get_bearer_token_from_credentials()
                if token:
                    self.bearer_token = token
                    self._initialized = True
                    logger.info("Nyckel service initialized with client credentials")
            except Exception as exc:
                logger.error(f"Failed to get Nyckel bearer token: {exc}")
        
        if not self._initialized:
            logger.warning("Nyckel credentials not set. Use NYCKEL_BEARER_TOKEN or NYCKEL_CLIENT_ID + NYCKEL_CLIENT_SECRET.")

    def _get_bearer_token_from_credentials(self) -> str:
        """Get access token from Nyckel connect/token (client_credentials grant)."""
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        resp = requests.post(
            NYCKEL_TOKEN_URL,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10,
        )
        resp.raise_for_status()
        result = resp.json()
        token = result.get("access_token")
        expires_in = int(result.get("expires_in", 3600))
        if token:
            self._token_expires_at = time.time() + expires_in - 60  # refresh 1 min early
        return token or ""

    def _ensure_token(self) -> bool:
        """If using client credentials, refresh token when expired. Returns True if we have a valid token."""
        if not self.client_id or not self.client_secret:
            return bool(self.bearer_token)
        if time.time() >= self._token_expires_at:
            try:
                self.bearer_token = self._get_bearer_token_from_credentials()
            except Exception as exc:
                logger.error(f"Nyckel token refresh failed: {exc}")
                return False
        return bool(self.bearer_token)

    # ---------- Public API ----------

    def detect_emotion_from_image(self, image_data: bytes) -> Dict[str, Any]:
        """
        Detect pet emotion from image bytes using Nyckel API.
        
        Strategy:
        - Convert image bytes to base64 data URL
        - Call Nyckel API with base64 image
        - Map Nyckel labels (Happy, Sad, Angry, Relaxed) to our emotion labels
        """
        if not self._initialized:
            return self._fallback_response("nyckel_not_initialized")
        if not self._ensure_token():
            return self._fallback_response("nyckel_no_token")

        try:
            # Convert image bytes to base64 data URL
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            # Detect image format from first bytes
            if image_data[:2] == b'\xff\xd8':
                mime_type = "image/jpeg"
            elif image_data[:8] == b'\x89PNG\r\n\x1a\n':
                mime_type = "image/png"
            elif image_data[:6] in [b'GIF87a', b'GIF89a']:
                mime_type = "image/gif"
            else:
                mime_type = "image/jpeg"  # Default
            
            data_url = f"data:{mime_type};base64,{image_base64}"
            
            # Call Nyckel API
            headers = {
                "Authorization": f"Bearer {self.bearer_token}",
                "Content-Type": "application/json",
            }
            payload = {"data": data_url}
            # Request top 5 labels for top_emotions
            invoke_url = f"{self.api_url}?labelCount=5"
            
            logger.info("Calling Nyckel API for dog emotion detection...")
            response = requests.post(invoke_url, json=payload, headers=headers, timeout=10)
            
            if response.status_code != 200:
                logger.error(f"Nyckel API error: {response.status_code} - {response.text}")
                print(f"Nyckel API error: {response.status_code} - {response.text}")
                return self._fallback_response(f"nyckel_api_error_{response.status_code}")
            
            result = response.json()

            # Print raw Nyckel response to dev server console for debugging
            try:
                logger.info(f"Nyckel raw response: {result}")
                # print(f"Nyckel raw response: {result}")
            except Exception:
                pass
            
            # Parse Nyckel response
            # Expected format: {\"labelName\": \"Happy\", \"confidence\": 0.95, ...}
            return self._parse_nyckel_response(result)
            
        except Exception as exc:
            logger.error(f"Nyckel API exception: {exc}")
            return self._fallback_response("nyckel_exception")

    def _parse_nyckel_response(self, result: dict) -> Dict[str, Any]:
        """Parse Nyckel API response and map to our emotion format."""
        # Nyckel returns: labelName, labelId, confidence, and optionally labelConfidences (when labelCount is set)
        if "labelName" not in result:
            return self._fallback_response("nyckel_no_predictions")
        
        nyckel_label = result.get("labelName", "").strip().lower()
        confidence = float(result.get("confidence", 0.5))
        
        # Map Nyckel labels to our emotion labels (Nyckel: Happy, Sad, Angry, Relaxed)
        emotion_mapping = {
            "happy": "happy",
            "sad": "sad",
            "angry": "angry",
            "relaxed": "calm",
        }
        
        pet_emotion = emotion_mapping.get(nyckel_label, "calm")
        
        # Build top_emotions from labelConfidences if present, else use main prediction
        top_emotions = []
        label_confidences = result.get("labelConfidences") or []
        for item in label_confidences[:5]:
            name = (item.get("labelName") or "").strip().lower()
            conf = float(item.get("confidence", 0.0))
            mapped = emotion_mapping.get(name, "calm")
            top_emotions.append({"emotion": mapped, "confidence": round(conf, 2)})
        
        if not top_emotions:
            top_emotions = [{"emotion": pet_emotion, "confidence": round(confidence, 2)}]
        
        return {
            "emotion": pet_emotion,
            "confidence": round(confidence, 2),
            "top_emotions": top_emotions,
            "ai_detector_type": "nyckel",
            "analysis_method": "nyckel_emotions_identifier",
            "model_architecture": "Nyckel Pretrained Emotions Classifier",
            "expected_accuracy": "high (pretrained on dog emotions)",
        }

    @staticmethod
    def _fallback_response(reason: str) -> Dict[str, Any]:
        """Fallback response when Nyckel API fails."""
        return {
            "emotion": "calm",
            "confidence": 0.5,
            "top_emotions": [{"emotion": "calm", "confidence": 0.5}],
            "ai_detector_type": "nyckel_fallback",
            "analysis_method": f"fallback_{reason}",
            "model_architecture": "Nyckel Pretrained Dog Emotions Classifier",
            "expected_accuracy": "unknown",
        }


# Global instance
nyckel_service = NyckelService()
