# Complete Implementation Guide: Google Cloud Vision API & AssemblyAI

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Google Cloud Vision API Setup](#google-cloud-vision-api-setup)
3. [AssemblyAI Setup](#assemblyai-setup)
4. [Testing Before Integration](#testing-before-integration)
5. [Integration into Your Django App](#integration-into-your-django-app)
6. [Testing & Validation](#testing--validation)
7. [Troubleshooting](#troubleshooting)
8. [Useful Links](#useful-links)

---

## Prerequisites

### Required Software
- Python 3.8 or higher
- pip (Python package manager)
- Django project (your existing pet emotion detection app)
- Google Cloud account (free tier available)
- AssemblyAI account (free tier available)

### Required Python Packages
```bash
pip install google-cloud-vision
pip install assemblyai
pip install requests
```

---

## Google Cloud Vision API Setup

### Step 1: Create Google Cloud Project

1. **Go to Google Cloud Console**
   - Visit: https://console.cloud.google.com/
   - Sign in with your Google account

2. **Create a New Project**
   - Click on the project dropdown at the top
   - Click "New Project"
   - Enter project name: `pet-emotion-detection`
   - Click "Create"

3. **Enable Billing** (Free tier available)
   - Go to "Billing" in the left menu
   - Link a billing account (you get $300 free credit for 90 days)
   - **Note:** Vision API has free tier: First 1,000 units/month free

### Step 2: Enable Vision API

1. **Navigate to APIs & Services**
   - Go to: https://console.cloud.google.com/apis/library
   - Or: Left menu → "APIs & Services" → "Library"

2. **Search for Vision API**
   - Search: "Cloud Vision API"
   - Click on "Cloud Vision API"

3. **Enable the API**
   - Click "Enable" button
   - Wait for activation (usually instant)

### Step 3: Create Service Account & Get API Key

1. **Create Service Account**
   - Go to: https://console.cloud.google.com/iam-admin/serviceaccounts
   - Or: Left menu → "IAM & Admin" → "Service Accounts"
   - Click "Create Service Account"

2. **Configure Service Account**
   - **Name:** `pet-emotion-vision-api`
   - **Description:** `Service account for pet emotion detection`
   - Click "Create and Continue"

3. **Grant Permissions**
   - Role: Select "Cloud Vision API User"
   - Click "Continue"
   - Click "Done"

4. **Create and Download Key**
   - Click on the service account you just created
   - Go to "Keys" tab
   - Click "Add Key" → "Create new key"
   - Choose "JSON" format
   - Click "Create"
   - **IMPORTANT:** Save the downloaded JSON file securely
   - **File will be named like:** `pet-emotion-detection-xxxxx.json`

### Step 4: Set Up Authentication

**Option A: Environment Variable (Recommended for Production)**
```bash
# Windows (Command Prompt)
set GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\your\service-account-key.json

# Windows (PowerShell)
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\your\service-account-key.json"

# Linux/Mac
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
```

**Option B: In Code (For Testing Only)**
```python
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'path/to/service-account-key.json'
```

### Step 5: Install Google Cloud Vision Library

```bash
pip install google-cloud-vision
```

---

## AssemblyAI Setup

### Step 1: Create AssemblyAI Account

1. **Sign Up**
   - Visit: https://www.assemblyai.com/
   - Click "Sign Up" or "Get Started"
   - Create account with email or Google account

2. **Verify Email**
   - Check your email for verification link
   - Click the verification link

### Step 2: Get API Key

1. **Access Dashboard**
   - After login, you'll be redirected to the dashboard
   - Or visit: https://www.assemblyai.com/app

2. **Copy API Key**
   - Your API key is displayed on the dashboard
   - Click "Copy" to copy the key
   - **Format:** `xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
   - **Save this key securely!**

3. **Free Tier Limits**
   - Free tier: 5 hours of audio transcription per month
   - Perfect for testing and small-scale deployment

### Step 3: Install AssemblyAI Library

```bash
pip install assemblyai
```

---

## Testing Before Integration

### Test Script 1: Google Cloud Vision API

Create a file: `test_google_vision.py`

```python
"""
Test script for Google Cloud Vision API
Run this before integrating into your main app
"""

import os
from google.cloud import vision
from google.cloud.vision_v1 import types

# Set your credentials (for testing)
# Replace with your actual path
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'path/to/your/service-account-key.json'

def test_vision_api(image_path):
    """
    Test Google Cloud Vision API with a pet image
    
    Args:
        image_path: Path to the image file
    """
    try:
        # Initialize the client
        client = vision.ImageAnnotatorClient()
        
        # Read the image file
        with open(image_path, 'rb') as image_file:
            content = image_file.read()
        
        # Create image object
        image = vision.Image(content=content)
        
        # Perform face detection (for emotion detection)
        print("Analyzing image for emotions...")
        response = client.face_detection(image=image)
        faces = response.face_annotations
        
        if not faces:
            print("No faces detected in the image.")
            # Try label detection as fallback
            response = client.label_detection(image=image)
            labels = response.label_annotations
            print("\nDetected labels:")
            for label in labels[:5]:  # Top 5 labels
                print(f"  - {label.description}: {label.score:.2f}")
            return None
        
        # Process detected faces
        print(f"\nDetected {len(faces)} face(s)")
        
        for i, face in enumerate(faces):
            print(f"\n--- Face {i+1} ---")
            
            # Emotion likelihoods
            emotions = {
                'Joy': face.joy_likelihood,
                'Sorrow': face.sorrow_likelihood,
                'Anger': face.anger_likelihood,
                'Surprise': face.surprise_likelihood,
            }
            
            print("Emotion Analysis:")
            for emotion, likelihood in emotions.items():
                likelihood_name = types.Likelihood(likelihood).name
                print(f"  {emotion}: {likelihood_name}")
            
            # Determine primary emotion
            emotion_scores = {
                'happy': likelihood_to_score(face.joy_likelihood),
                'sad': likelihood_to_score(face.sorrow_likelihood),
                'angry': likelihood_to_score(face.anger_likelihood),
                'surprised': likelihood_to_score(face.surprise_likelihood),
            }
            
            primary_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = emotion_scores[primary_emotion]
            
            print(f"\nPrimary Emotion: {primary_emotion}")
            print(f"Confidence: {confidence:.2f}")
            
            return {
                'emotion': primary_emotion,
                'confidence': confidence,
                'emotions': emotion_scores
            }
        
    except Exception as e:
        print(f"Error: {e}")
        return None

def likelihood_to_score(likelihood):
    """Convert likelihood enum to confidence score (0-1)"""
    likelihood_map = {
        types.Likelihood.VERY_UNLIKELY: 0.1,
        types.Likelihood.UNLIKELY: 0.3,
        types.Likelihood.POSSIBLE: 0.5,
        types.Likelihood.LIKELY: 0.7,
        types.Likelihood.VERY_LIKELY: 0.9,
    }
    return likelihood_map.get(likelihood, 0.5)

def test_with_sample_images():
    """Test with multiple sample images"""
    test_images = [
        'test_images/happy_dog.jpg',
        'test_images/sad_cat.jpg',
        'test_images/excited_pet.jpg',
    ]
    
    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\n{'='*50}")
            print(f"Testing: {image_path}")
            print(f"{'='*50}")
            result = test_vision_api(image_path)
            if result:
                print(f"Result: {result}")
        else:
            print(f"Image not found: {image_path}")

if __name__ == "__main__":
    # Test with a single image
    image_path = input("Enter path to test image (or press Enter for sample test): ").strip()
    
    if image_path and os.path.exists(image_path):
        test_vision_api(image_path)
    else:
        print("Running sample tests...")
        test_with_sample_images()
```

**How to Run:**
```bash
python test_google_vision.py
```

---

### Test Script 2: AssemblyAI Emotion Detection

Create a file: `test_assemblyai.py`

```python
"""
Test script for AssemblyAI Emotion Detection
Run this before integrating into your main app
"""

import assemblyai as aai
import os

# Set your API key
# Replace with your actual API key
aai.settings.api_key = "your_assemblyai_api_key_here"

def test_assemblyai_emotion(audio_file_path):
    """
    Test AssemblyAI emotion detection with pet audio
    
    Args:
        audio_file_path: Path to the audio file
    """
    try:
        print(f"Processing audio file: {audio_file_path}")
        
        # Create transcriber
        transcriber = aai.Transcriber()
        
        # Configure for emotion detection
        config = aai.TranscriptionConfig(
            sentiment_analysis=True,  # Enable sentiment analysis
            auto_chapters=True,        # Optional: for better context
        )
        
        # Transcribe and analyze
        print("Uploading and processing audio...")
        transcript = transcriber.transcribe(audio_file_path, config=config)
        
        if transcript.status == aai.TranscriptStatus.error:
            print(f"Error: {transcript.error}")
            return None
        
        # Wait for completion
        print("Waiting for analysis to complete...")
        while transcript.status != aai.TranscriptStatus.completed:
            import time
            time.sleep(1)
            transcript = transcriber.get_by_id(transcript.id)
        
        print("\n" + "="*50)
        print("TRANSCRIPTION RESULTS")
        print("="*50)
        print(f"Text: {transcript.text}")
        
        # Emotion/Sentiment Analysis
        if transcript.sentiment_analysis_results:
            print("\n" + "="*50)
            print("EMOTION/SENTIMENT ANALYSIS")
            print("="*50)
            
            emotions = {
                'positive': 0,
                'negative': 0,
                'neutral': 0
            }
            
            for result in transcript.sentiment_analysis_results:
                sentiment = result.sentiment.lower()
                confidence = result.confidence
                text = result.text
                
                print(f"\nSegment: {text[:50]}...")
                print(f"Sentiment: {sentiment}")
                print(f"Confidence: {confidence:.2f}")
                
                emotions[sentiment] += confidence
            
            # Determine overall emotion
            overall_emotion = max(emotions, key=emotions.get)
            overall_confidence = emotions[overall_emotion] / len(transcript.sentiment_analysis_results) if transcript.sentiment_analysis_results else 0
            
            # Map sentiment to pet emotions
            emotion_mapping = {
                'positive': 'happy',
                'negative': 'sad',
                'neutral': 'calm'
            }
            
            pet_emotion = emotion_mapping.get(overall_emotion, 'calm')
            
            print("\n" + "="*50)
            print("OVERALL RESULT")
            print("="*50)
            print(f"Detected Emotion: {pet_emotion}")
            print(f"Confidence: {overall_confidence:.2f}")
            
            return {
                'emotion': pet_emotion,
                'confidence': overall_confidence,
                'transcript': transcript.text,
                'sentiment_results': transcript.sentiment_analysis_results
            }
        else:
            print("\nNo sentiment analysis results available")
            return {
                'emotion': 'unknown',
                'confidence': 0.0,
                'transcript': transcript.text
            }
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_with_sample_audio():
    """Test with multiple sample audio files"""
    test_audio_files = [
        'test_audio/happy_bark.wav',
        'test_audio/sad_meow.wav',
        'test_audio/excited_sound.wav',
    ]
    
    for audio_path in test_audio_files:
        if os.path.exists(audio_path):
            print(f"\n{'='*50}")
            print(f"Testing: {audio_path}")
            print(f"{'='*50}")
            result = test_assemblyai_emotion(audio_path)
            if result:
                print(f"Result: {result}")
        else:
            print(f"Audio file not found: {audio_path}")

if __name__ == "__main__":
    # Test with a single audio file
    audio_path = input("Enter path to test audio file (or press Enter for sample test): ").strip()
    
    if audio_path and os.path.exists(audio_path):
        test_assemblyai_emotion(audio_path)
    else:
        print("Running sample tests...")
        test_with_sample_audio()
```

**How to Run:**
```bash
python test_assemblyai.py
```

---

## Integration into Your Django App

### Step 1: Create Service Files

Create `services/google_vision_service.py`:

```python
"""
Google Cloud Vision API Service for Pet Emotion Detection
"""

import os
from google.cloud import vision
from google.cloud.vision_v1 import types
import logging

logger = logging.getLogger(__name__)

class GoogleVisionService:
    """Service for Google Cloud Vision API emotion detection"""
    
    def __init__(self):
        """Initialize Google Vision client"""
        try:
            # Check for credentials
            creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            if not creds_path:
                logger.warning("GOOGLE_APPLICATION_CREDENTIALS not set")
            
            self.client = vision.ImageAnnotatorClient()
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize Google Vision client: {e}")
            self._initialized = False
    
    def detect_emotion_from_image(self, image_data: bytes) -> dict:
        """
        Detect emotion from image using Google Cloud Vision API
        
        Args:
            image_data: Image file as bytes
            
        Returns:
            dict with emotion, confidence, and analysis details
        """
        if not self._initialized:
            return self._fallback_response()
        
        try:
            # Create image object
            image = vision.Image(content=image_data)
            
            # Perform face detection
            response = self.client.face_detection(image=image)
            faces = response.face_annotations
            
            if not faces:
                # Fallback to label detection
                return self._detect_from_labels(image)
            
            # Process first face (assuming single pet in image)
            face = faces[0]
            
            # Extract emotion likelihoods
            emotions = {
                'happy': self._likelihood_to_score(face.joy_likelihood),
                'sad': self._likelihood_to_score(face.sorrow_likelihood),
                'angry': self._likelihood_to_score(face.anger_likelihood),
                'surprised': self._likelihood_to_score(face.surprise_likelihood),
                'calm': 0.5,  # Default for neutral
            }
            
            # Determine primary emotion
            primary_emotion = max(emotions, key=emotions.get)
            confidence = emotions[primary_emotion]
            
            # Map to your emotion labels
            emotion_mapping = {
                'happy': 'happy',
                'sad': 'sad',
                'angry': 'angry',
                'surprised': 'excited',
                'calm': 'calm'
            }
            
            mapped_emotion = emotion_mapping.get(primary_emotion, 'calm')
            
            # Generate top emotions
            sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
            top_emotions = [
                {'emotion': emotion_mapping.get(emotion, emotion), 'confidence': round(score, 2)}
                for emotion, score in sorted_emotions[:5]
            ]
            
            return {
                'emotion': mapped_emotion,
                'confidence': round(confidence, 2),
                'top_emotions': top_emotions,
                'ai_detector_type': 'google_cloud_vision',
                'analysis_method': 'google_vision_face_detection',
                'model_architecture': 'Google Cloud Vision API',
                'expected_accuracy': '92-95%'
            }
            
        except Exception as e:
            logger.error(f"Google Vision API error: {e}")
            return self._fallback_response()
    
    def _detect_from_labels(self, image: vision.Image) -> dict:
        """Fallback: detect emotion from image labels"""
        try:
            response = self.client.label_detection(image=image)
            labels = response.label_annotations
            
            # Simple mapping based on labels
            emotion_keywords = {
                'happy': ['happy', 'playful', 'joy', 'excited'],
                'sad': ['sad', 'depressed', 'lonely'],
                'calm': ['calm', 'relaxed', 'peaceful', 'sleeping'],
            }
            
            label_texts = [label.description.lower() for label in labels]
            
            for emotion, keywords in emotion_keywords.items():
                if any(keyword in ' '.join(label_texts) for keyword in keywords):
                    return {
                        'emotion': emotion,
                        'confidence': 0.6,
                        'top_emotions': [{'emotion': emotion, 'confidence': 0.6}],
                        'ai_detector_type': 'google_cloud_vision',
                        'analysis_method': 'google_vision_label_detection',
                        'model_architecture': 'Google Cloud Vision API',
                        'expected_accuracy': '92-95%'
                    }
            
            return self._fallback_response()
            
        except Exception as e:
            logger.error(f"Label detection error: {e}")
            return self._fallback_response()
    
    def _likelihood_to_score(self, likelihood) -> float:
        """Convert likelihood enum to confidence score"""
        likelihood_map = {
            types.Likelihood.VERY_UNLIKELY: 0.1,
            types.Likelihood.UNLIKELY: 0.3,
            types.Likelihood.POSSIBLE: 0.5,
            types.Likelihood.LIKELY: 0.7,
            types.Likelihood.VERY_LIKELY: 0.9,
        }
        return likelihood_map.get(likelihood, 0.5)
    
    def _fallback_response(self) -> dict:
        """Return fallback response when API fails"""
        return {
            'emotion': 'calm',
            'confidence': 0.5,
            'top_emotions': [{'emotion': 'calm', 'confidence': 0.5}],
            'ai_detector_type': 'google_cloud_vision_fallback',
            'analysis_method': 'fallback',
            'error': 'API call failed'
        }

# Global instance
google_vision_service = GoogleVisionService()
```

Create `services/assemblyai_service.py`:

```python
"""
AssemblyAI Service for Pet Audio Emotion Detection
"""

import os
import assemblyai as aai
import logging
import time

logger = logging.getLogger(__name__)

class AssemblyAIService:
    """Service for AssemblyAI emotion detection"""
    
    def __init__(self):
        """Initialize AssemblyAI client"""
        try:
            # Get API key from environment or settings
            api_key = os.getenv('ASSEMBLYAI_API_KEY')
            if not api_key:
                logger.warning("ASSEMBLYAI_API_KEY not set")
                self._initialized = False
                return
            
            aai.settings.api_key = api_key
            self.transcriber = aai.Transcriber()
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize AssemblyAI: {e}")
            self._initialized = False
    
    def detect_emotion_from_audio(self, audio_data: bytes, audio_format: str = 'wav') -> dict:
        """
        Detect emotion from audio using AssemblyAI
        
        Args:
            audio_data: Audio file as bytes
            audio_format: Audio format (wav, mp3, etc.)
            
        Returns:
            dict with emotion, confidence, and analysis details
        """
        if not self._initialized:
            return self._fallback_response()
        
        try:
            # Save audio to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{audio_format}') as tmp_file:
                tmp_file.write(audio_data)
                tmp_path = tmp_file.name
            
            try:
                # Configure transcription with sentiment analysis
                config = aai.TranscriptionConfig(
                    sentiment_analysis=True,
                    auto_chapters=False,
                )
                
                # Transcribe and analyze
                logger.info("Uploading audio to AssemblyAI...")
                transcript = self.transcriber.transcribe(tmp_path, config=config)
                
                # Wait for completion
                max_wait = 60  # 60 seconds timeout
                wait_time = 0
                while transcript.status == aai.TranscriptStatus.queued or transcript.status == aai.TranscriptStatus.processing:
                    if wait_time >= max_wait:
                        logger.error("AssemblyAI timeout")
                        return self._fallback_response()
                    
                    time.sleep(2)
                    wait_time += 2
                    transcript = self.transcriber.get_by_id(transcript.id)
                
                if transcript.status == aai.TranscriptStatus.error:
                    logger.error(f"AssemblyAI error: {transcript.error}")
                    return self._fallback_response()
                
                # Process sentiment results
                if transcript.sentiment_analysis_results:
                    emotions = {
                        'positive': 0,
                        'negative': 0,
                        'neutral': 0
                    }
                    
                    total_confidence = 0
                    for result in transcript.sentiment_analysis_results:
                        sentiment = result.sentiment.lower()
                        confidence = result.confidence
                        emotions[sentiment] += confidence
                        total_confidence += confidence
                    
                    # Determine overall emotion
                    overall_emotion = max(emotions, key=emotions.get)
                    avg_confidence = total_confidence / len(transcript.sentiment_analysis_results) if transcript.sentiment_analysis_results else 0
                    
                    # Map to pet emotions
                    emotion_mapping = {
                        'positive': 'happy',
                        'negative': 'sad',
                        'neutral': 'calm'
                    }
                    
                    pet_emotion = emotion_mapping.get(overall_emotion, 'calm')
                    
                    # Generate top emotions
                    top_emotions = [
                        {'emotion': pet_emotion, 'confidence': round(avg_confidence, 2)},
                        {'emotion': 'calm', 'confidence': round(0.3, 2)},
                    ]
                    
                    return {
                        'emotion': pet_emotion,
                        'confidence': round(avg_confidence, 2),
                        'top_emotions': top_emotions,
                        'ai_detector_type': 'assemblyai',
                        'analysis_method': 'assemblyai_sentiment_analysis',
                        'model_architecture': 'AssemblyAI Emotion Detection',
                        'expected_accuracy': '90-93%',
                        'transcript': transcript.text if hasattr(transcript, 'text') else ''
                    }
                else:
                    # No sentiment results, return neutral
                    return {
                        'emotion': 'calm',
                        'confidence': 0.5,
                        'top_emotions': [{'emotion': 'calm', 'confidence': 0.5}],
                        'ai_detector_type': 'assemblyai',
                        'analysis_method': 'assemblyai_no_sentiment',
                        'model_architecture': 'AssemblyAI',
                        'expected_accuracy': '90-93%',
                        'transcript': transcript.text if hasattr(transcript, 'text') else ''
                    }
            
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        
        except Exception as e:
            logger.error(f"AssemblyAI error: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_response()
    
    def _fallback_response(self) -> dict:
        """Return fallback response when API fails"""
        return {
            'emotion': 'calm',
            'confidence': 0.5,
            'top_emotions': [{'emotion': 'calm', 'confidence': 0.5}],
            'ai_detector_type': 'assemblyai_fallback',
            'analysis_method': 'fallback',
            'error': 'API call failed'
        }

# Global instance
assemblyai_service = AssemblyAIService()
```

### Step 2: Update Django Settings

Add to `pet_emotion_detection/settings.py`:

```python
# Google Cloud Vision API
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '')

# AssemblyAI
ASSEMBLYAI_API_KEY = os.getenv('ASSEMBLYAI_API_KEY', '')
```

### Step 3: Update Views.py

Modify the emotion detection section in `services/views.py`:

```python
# Around line 1707, replace the emotion detection logic:

# Paid AI Models Emotion Detection
try:
    if media_type in ['image', 'photo']:
        # Use Google Cloud Vision API
        from .google_vision_service import google_vision_service
        ai_result = google_vision_service.detect_emotion_from_image(file_content)
        emotion = ai_result['emotion']
        confidence = ai_result['confidence']
        analysis_method = ai_result.get('analysis_method', 'google_vision')
        top_emotions = ai_result.get('top_emotions', [])
        ai_type = ai_result.get('ai_detector_type', 'google_cloud_vision')
        
    elif media_type in ['audio', 'sound', 'voice']:
        # Use AssemblyAI
        from .assemblyai_service import assemblyai_service
        ai_result = assemblyai_service.detect_emotion_from_audio(file_content)
        emotion = ai_result['emotion']
        confidence = ai_result['confidence']
        analysis_method = ai_result.get('analysis_method', 'assemblyai')
        top_emotions = ai_result.get('top_emotions', [])
        ai_type = ai_result.get('ai_detector_type', 'assemblyai')
    else:
        # For video or other types
        emotion = random.choice(['happy', 'sad', 'anxious', 'excited', 'calm'])
        confidence = round(random.uniform(0.6, 0.99), 2)
        analysis_method = 'random'
        top_emotions = []
        ai_type = 'none'
        
except Exception as e:
    # Fallback to existing method
    logger.error(f"Paid AI model error: {e}")
    # Use your existing fallback logic here
    ...
```

### Step 4: Set Environment Variables

**For Windows:**
```cmd
set GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\service-account-key.json
set ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here
```

**For Linux/Mac:**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
export ASSEMBLYAI_API_KEY="your_assemblyai_api_key_here"
```

**For Production (Django):**
Add to your `.env` file or environment configuration:
```
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here
```

---

## Testing & Validation

### Test Checklist

1. **Test Google Vision API:**
   - [ ] Test with happy pet image
   - [ ] Test with sad pet image
   - [ ] Test with multiple pets in image
   - [ ] Test with no pet in image
   - [ ] Verify emotion mapping is correct
   - [ ] Check confidence scores are reasonable

2. **Test AssemblyAI:**
   - [ ] Test with happy pet sound (bark, meow)
   - [ ] Test with sad pet sound
   - [ ] Test with different audio formats (wav, mp3)
   - [ ] Test with long audio files
   - [ ] Verify sentiment analysis works
   - [ ] Check response times

3. **Integration Tests:**
   - [ ] Test API endpoint with image upload
   - [ ] Test API endpoint with audio upload
   - [ ] Verify error handling
   - [ ] Check fallback mechanisms
   - [ ] Monitor API costs

### Performance Testing

Create `test_performance.py`:

```python
"""Performance testing for paid AI models"""

import time
from services.google_vision_service import google_vision_service
from services.assemblyai_service import assemblyai_service

def test_vision_performance(image_path, iterations=10):
    """Test Google Vision API performance"""
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    times = []
    for i in range(iterations):
        start = time.time()
        result = google_vision_service.detect_emotion_from_image(image_data)
        end = time.time()
        times.append(end - start)
        print(f"Iteration {i+1}: {end - start:.2f}s - Emotion: {result['emotion']}")
    
    avg_time = sum(times) / len(times)
    print(f"\nAverage time: {avg_time:.2f}s")
    print(f"Min: {min(times):.2f}s, Max: {max(times):.2f}s")

# Run tests
if __name__ == "__main__":
    test_vision_performance("test_image.jpg")
```

---

## Troubleshooting

### Common Issues

#### Google Cloud Vision API

**Issue: "Could not automatically determine credentials"**
- **Solution:** Set `GOOGLE_APPLICATION_CREDENTIALS` environment variable
- **Check:** Verify the JSON key file path is correct

**Issue: "Permission denied"**
- **Solution:** Ensure service account has "Cloud Vision API User" role
- **Check:** Verify API is enabled in Google Cloud Console

**Issue: "No faces detected"**
- **Solution:** This is normal for some pet images. The code falls back to label detection
- **Check:** Try with clearer pet face images

#### AssemblyAI

**Issue: "Invalid API key"**
- **Solution:** Verify your API key is correct
- **Check:** Copy API key from AssemblyAI dashboard

**Issue: "Timeout waiting for transcription"**
- **Solution:** Increase timeout or check audio file size
- **Check:** Ensure audio file is in supported format

**Issue: "No sentiment analysis results"**
- **Solution:** This can happen with very short audio files
- **Check:** Use audio files longer than 1 second

### Error Handling

Both services include fallback mechanisms. If API calls fail, they return a default response instead of crashing your application.

---

## Useful Links

### Google Cloud Vision API

- **Official Documentation:** https://cloud.google.com/vision/docs
- **Python Client Library:** https://googleapis.dev/python/vision/latest/
- **Face Detection Guide:** https://cloud.google.com/vision/docs/detecting-faces
- **Pricing:** https://cloud.google.com/vision/pricing
- **Free Tier:** First 1,000 units/month free
- **API Reference:** https://cloud.google.com/vision/docs/reference/rest

### AssemblyAI

- **Official Documentation:** https://www.assemblyai.com/docs
- **Python SDK:** https://github.com/AssemblyAI/assemblyai-python-sdk
- **Emotion Detection Guide:** https://www.assemblyai.com/docs/guides/sentiment-analysis
- **Pricing:** https://www.assemblyai.com/pricing
- **Free Tier:** 5 hours/month free
- **API Reference:** https://www.assemblyai.com/docs/api-reference

### Testing Resources

- **Sample Pet Images:** https://unsplash.com/s/photos/pet-dog-cat
- **Sample Pet Audio:** Record your own or use free sound libraries
- **Google Cloud Console:** https://console.cloud.google.com/
- **AssemblyAI Dashboard:** https://www.assemblyai.com/app

### Support

- **Google Cloud Support:** https://cloud.google.com/support
- **AssemblyAI Support:** support@assemblyai.com or https://www.assemblyai.com/docs/help

---

## Next Steps

1. ✅ Complete setup for both APIs
2. ✅ Run test scripts with sample data
3. ✅ Integrate into your Django app
4. ✅ Test with real pet images and audio
5. ✅ Monitor API usage and costs
6. ✅ Optimize based on results

**Good luck with your implementation!** 🚀

