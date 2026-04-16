# AI Package Dependencies

## 📦 Packages Used in Scan API

### **Image AI (`advanced_image_ai.py`)**
- `tensorflow` - Deep learning framework
- `numpy` - Numerical computing
- `opencv-python` - Image processing (face detection, filtering)
- `Pillow` - Image manipulation (resize, enhance, filters)
- `scikit-learn` - Machine learning utilities

### **Audio AI (`advanced_audio_ai.py`)**
- `tensorflow` - Deep learning framework
- `numpy` - Numerical computing
- `librosa` - Audio analysis (MFCC, spectral features)
- `soundfile` - Audio I/O operations
- `standard-aifc` - Audio format support

---

## 📋 Complete Requirements File

**File:** `requirements.txt`

All packages needed for:
- ✅ Django API
- ✅ Firebase integration
- ✅ AI image processing
- ✅ AI audio processing
- ✅ Twilio SMS (optional)
- ✅ Vonage SMS support

---

## 🔧 Installation

```bash
pip install -r requirements.txt
```

**Or if you're using venv:**
```bash
venv\Scripts\activate
pip install -r requirements.txt
```

---

## ✅ Packages Added to requirements.txt

```txt
# Twilio SMS Integration (Optional)
twilio==8.10.0

# AI/ML Dependencies for Pet Emotion Detection
tensorflow>=2.15.0
numpy>=1.24.0
opencv-python>=4.8.0
Pillow>=10.2.0
scikit-learn>=1.3.0
librosa>=0.11.0
soundfile>=0.13.0
standard-aifc>=3.13.0
```

---

## 📝 Summary

**Before:**
- ❌ `requirements.txt` - Django + Firebase
- ❌ `requirements-ai.txt` - AI packages

**After:**
- ✅ `requirements.txt` - Everything in one file!

**Deleted:**
- ✅ `requirements-ai.txt` removed

---

## 🎯 What Each Package Does

| Package | Used For | File |
|---------|----------|------|
| `tensorflow` | AI model inference | Both |
| `numpy` | Array operations | Both |
| `opencv-python` | Face detection, image processing | Image AI |
| `Pillow` | Image manipulation | Image AI |
| `librosa` | Audio feature extraction | Audio AI |
| `soundfile` | Audio file I/O | Audio AI |
| `scikit-learn` | ML utilities | Image AI |
| `standard-aifc` | Audio format support | Audio AI |

---

**All dependencies are now in one file: `requirements.txt`**
