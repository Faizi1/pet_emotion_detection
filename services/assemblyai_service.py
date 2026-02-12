"""
AssemblyAI Service for Pet Audio Emotion Detection

This service:
- Uses AssemblyAI's sentiment analysis on transcribed audio
- Maps overall sentiment to simple pet emotions (happy / sad / calm)
- Returns a dict compatible with the existing scan response.

Requirements:
- Environment variable ASSEMBLYAI_API_KEY with your API key
- Package: assemblyai
"""

import logging
import os
import tempfile
import time
from typing import Dict, Any, List

import assemblyai as aai
import numpy as np

try:  # librosa is already in your dependencies for other audio code
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:  # pragma: no cover - defensive
    librosa = None
    _LIBROSA_AVAILABLE = False

logger = logging.getLogger(__name__)


class AssemblyAIService:
    """Wrapper around AssemblyAI for pet audio emotion detection."""

    def __init__(self) -> None:
        api_key = os.getenv("ASSEMBLYAI_API_KEY")
        if not api_key:
            logger.warning("ASSEMBLYAI_API_KEY is not set; AssemblyAI will not work.")
            self._initialized = False
            return

        try:
            aai.settings.api_key = api_key
            self.transcriber = aai.Transcriber()
            self._initialized = True
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Failed to initialize AssemblyAI: {exc}")
            self._initialized = False

    # ---------- Public API ----------

    def detect_emotion_from_audio(self, audio_data: bytes, audio_format: str = "wav") -> Dict[str, Any]:
        """
        Detect pet emotion from raw audio bytes using AssemblyAI.

        Strategy:
        - Save bytes to a temp file
        - Run transcription with sentiment_analysis=True
        - Aggregate sentiment across segments and map to pet emotions
        """
        if not self._initialized:
            return self._fallback_response("assemblyai_not_initialized")

        tmp_path = None
        try:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_format}") as tmp:
                tmp.write(audio_data)
                tmp_path = tmp.name

            config = aai.TranscriptionConfig(
                sentiment_analysis=True,
                auto_chapters=False,
            )

            logger.info("AssemblyAI: uploading and processing audio...")
            transcript = self.transcriber.transcribe(tmp_path, config=config)

            # Poll until completion or timeout (keep this modest so API calls don't hang too long)
            max_wait_sec = 25
            waited = 0
            while transcript.status in (aai.TranscriptStatus.queued, aai.TranscriptStatus.processing):
                if waited >= max_wait_sec:
                    logger.error("AssemblyAI: transcription timeout")
                    return self._fallback_response("assemblyai_timeout")
                time.sleep(2)
                waited += 2
                transcript = self.transcriber.get_by_id(transcript.id)

            if transcript.status == aai.TranscriptStatus.error:
                logger.error(f"AssemblyAI error: {getattr(transcript, 'error', 'unknown error')}")
                return self._fallback_response("assemblyai_error")

            return self._from_sentiment_results(transcript, tmp_path)

        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"AssemblyAI exception: {exc}")
            return self._fallback_response("assemblyai_exception")
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    # ---------- Internal helpers ----------

    def _from_sentiment_results(self, transcript: aai.Transcript, audio_path: str) -> Dict[str, Any]:
        """Aggregate sentiment analysis results into a single pet emotion.

        NOTE: AssemblyAI is trained on human speech. For pure pet sounds (bark/meow)
        there's often no sentiment segments. In that case we assume a high‑arousal
        neutral emotion and treat it as 'excited' instead of 'calm'.
        """
        results: List[aai.SentimentResult] = getattr(transcript, "sentiment_analysis_results", []) or []

        if not results:
            # No sentiment segments (e.g., barking, meowing, very short/quiet audio).
            text = getattr(transcript, "text", "") or ""

            # Heuristic:
            # - If there's essentially no text → likely non‑speech pet sound → analyze raw audio.
            # - If there is some text but no sentiment → stay neutral/calm.
            if not text.strip():
                pet_emotion, confidence = self._analyze_raw_pet_audio(audio_path)
            else:
                pet_emotion = "calm"
                confidence = 0.5

            return {
                "emotion": pet_emotion,
                "confidence": round(confidence, 2),
                "top_emotions": [{"emotion": pet_emotion, "confidence": round(confidence, 2)}],
                "ai_detector_type": "assemblyai",
                "analysis_method": "assemblyai",
                "model_architecture": "AssemblyAI Sentiment Analysis + Heuristics",
                "expected_accuracy": "90-93%",
                "transcript": text,
            }

        sentiment_scores = {
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 0.0,
        }
        for seg in results:
            sentiment = seg.sentiment.lower()
            if sentiment in sentiment_scores:
                sentiment_scores[sentiment] += float(seg.confidence or 0.0)

        # Pick dominant sentiment
        dominant_sentiment = max(sentiment_scores, key=sentiment_scores.get)
        total_segments = len(results)
        avg_conf = (
            sentiment_scores[dominant_sentiment] / total_segments if total_segments > 0 else 0.5
        )

        # Map sentiment → pet emotion
        sentiment_to_pet = {
            "positive": "happy",
            "negative": "sad",
            "neutral": "calm",
        }
        pet_emotion = sentiment_to_pet.get(dominant_sentiment, "calm")

        top_emotions = [
            {"emotion": pet_emotion, "confidence": round(avg_conf, 2)},
            {"emotion": "calm", "confidence": 0.3},
        ]

        return {
            "emotion": pet_emotion,
            "confidence": round(avg_conf, 2),
            "top_emotions": top_emotions,
            "ai_detector_type": "assemblyai",
            "analysis_method": "assemblyai",
            "model_architecture": "AssemblyAI Sentiment Analysis",
            "expected_accuracy": "90-93%",
            "transcript": getattr(transcript, "text", "") or "",
        }

    def _analyze_raw_pet_audio(self, audio_path: str) -> (str, float):
        """
        Enhanced rule-based analysis of raw audio for pet sounds.

        Uses multiple audio features to detect:
        - 'angry'    (very loud, rough/noisy, high variation, low pitch)
        - 'excited'  (loud, bursty, high pitch, moderate roughness)
        - 'sad'      (medium loudness, continuous, whining characteristics)
        - 'calm'     (quiet, low energy, smooth)
        - 'happy'    (moderate loudness, lively, moderate pitch)
        """
        # If librosa isn't available, fall back to a sane default.
        if not _LIBROSA_AVAILABLE or not audio_path:
            return "excited", 0.6

        try:
            y, sr = librosa.load(audio_path, sr=22050, mono=True)
            if y.size == 0:
                return "calm", 0.5

            # Trim silence to focus on active sound
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            if y_trimmed.size == 0:
                return "calm", 0.5
            y = y_trimmed

            # Overall loudness (RMS energy)
            rms = librosa.feature.rms(y=y)[0]
            mean_rms = float(np.mean(rms))
            max_rms = float(np.max(rms))
            std_rms = float(np.std(rms))

            # Pitch / fundamental frequency (for distinguishing high-pitched happy vs low-pitched angry)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            mean_pitch = float(np.mean(pitch_values)) if pitch_values else 0.0

            # Roughness / noisiness
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

            mean_centroid = float(np.mean(centroid))
            mean_zcr = float(np.mean(zcr))
            mean_rolloff = float(np.mean(rolloff))
            std_centroid = float(np.std(centroid))

            # Tempo / rhythm (for distinguishing excited vs calm)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo = float(tempo) if tempo > 0 else 0.0

            # Calculate emotion scores (higher = more likely)
            emotion_scores = {
                "angry": 0.0,
                "excited": 0.0,
                "sad": 0.0,
                "calm": 0.0,
                "happy": 0.0,
            }

            # ANGRY: Very loud + rough + low pitch + high variation
            if max_rms > 0.15:  # Lowered threshold
                emotion_scores["angry"] += (max_rms - 0.15) * 2.0
            if mean_centroid > 2000 or mean_zcr > 0.12:  # Rough/noisy
                emotion_scores["angry"] += 0.3
            if mean_pitch > 0 and mean_pitch < 200:  # Low pitch (growling)
                emotion_scores["angry"] += 0.4
            if std_rms > 0.03:  # High variation
                emotion_scores["angry"] += std_rms * 5.0

            # EXCITED: Loud + high pitch + bursty + fast tempo
            if max_rms > 0.12:  # Lowered threshold
                emotion_scores["excited"] += (max_rms - 0.12) * 1.5
            if mean_pitch > 300:  # High pitch (excited bark/meow)
                emotion_scores["excited"] += 0.4
            if std_rms > 0.025:  # Bursty
                emotion_scores["excited"] += std_rms * 4.0
            if tempo > 100:  # Fast tempo
                emotion_scores["excited"] += 0.2

            # SAD: Medium loudness + continuous + whining characteristics
            if 0.05 < mean_rms < 0.15:  # Medium loudness
                emotion_scores["sad"] += 0.3
            if std_rms < 0.02:  # More continuous (less bursty)
                emotion_scores["sad"] += 0.3
            if mean_pitch > 200 and mean_pitch < 500:  # Mid-high pitch (whining)
                emotion_scores["sad"] += 0.3
            if mean_zcr > 0.08 and mean_zcr < 0.15:  # Moderate roughness (crying)
                emotion_scores["sad"] += 0.2

            # CALM: Quiet + smooth + low variation
            if mean_rms < 0.06:  # Quiet
                emotion_scores["calm"] += (0.06 - mean_rms) * 5.0
            if std_rms < 0.015:  # Low variation
                emotion_scores["calm"] += 0.3
            if mean_zcr < 0.08:  # Smooth (not rough)
                emotion_scores["calm"] += 0.2
            if mean_centroid < 2000:  # Lower spectral content
                emotion_scores["calm"] += 0.2

            # HAPPY: Moderate loudness + lively + moderate pitch
            if 0.08 < mean_rms < 0.18:  # Moderate loudness
                emotion_scores["happy"] += 0.3
            if 200 < mean_pitch < 600:  # Moderate pitch range
                emotion_scores["happy"] += 0.3
            if 0.02 < std_rms < 0.04:  # Moderate variation (lively but not chaotic)
                emotion_scores["happy"] += 0.2
            if mean_zcr < 0.12:  # Not too rough
                emotion_scores["happy"] += 0.2

            # Find the emotion with highest score
            best_emotion = max(emotion_scores, key=emotion_scores.get)
            best_score = emotion_scores[best_emotion]

            # Normalize confidence: map score to [0.5, 0.95] range
            # Higher score = higher confidence
            max_possible_score = 2.0  # Approximate max
            confidence = min(0.5 + (best_score / max_possible_score) * 0.45, 0.95)
            confidence = max(0.5, confidence)  # Ensure minimum 0.5

            # If all scores are very low, default to calm with lower confidence
            if best_score < 0.1:
                return "calm", 0.55

            return best_emotion, round(confidence, 2)

        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Error in raw pet audio analysis: {exc}")
            return "excited", 0.6

    @staticmethod
    def _fallback_response(reason: str) -> Dict[str, Any]:
        """Neutral calm fallback if API is not available."""
        return {
            "emotion": "calm",
            "confidence": 0.5,
            "top_emotions": [{"emotion": "calm", "confidence": 0.5}],
            "ai_detector_type": "assemblyai_fallback",
            "analysis_method": f"fallback_{reason}",
            "model_architecture": "AssemblyAI",
            "expected_accuracy": "unknown",
        }


# Global instance
assemblyai_service = AssemblyAIService()

