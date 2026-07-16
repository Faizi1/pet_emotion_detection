# PetMood — AI Scan Upgrade & Production Deployment Update

**Date:** June 2026  
**Status:** Deployed to production  
**Production API:** https://ai-pet-mood.onrender.com

---

## Summary

The pet emotion scan system has been upgraded from a single-step external API to a **two-stage AI pipeline** running on our own server. The backend is now deployed on a **new Render server** with updated production configuration.

---

## What Changed — AI Model (Image Scans)

### Before
- Image uploaded → sent directly to Nyckel API
- No check if a pet was actually in the photo
- Non-pet images (people, rooms, objects) could still return an emotion
- Limited accuracy and no control over the model

### After (Two-Stage Pipeline)

![Pet detection and emotion analysis flow](Pet%20Detection%20Flow.png)

*Figure: Two-stage pipeline — YOLOv8 validates and crops the pet, then the emotion model runs via ONNX (or PyTorch); Nyckel is used only if local models are unavailable.*

**Step-by-step:**

| Step | What happens |
|------|----------------|
| 1 | User uploads a pet photo from the app |
| 2 | **Stage 1 — YOLOv8** checks if a cat or dog is in the image |
| 3 | If **no pet** → API returns **HTTP 422** (`no_pet_in_image`) so the app can ask the user to retake the photo |
| 4 | If **pet found** → the pet region is cropped (face + body) |
| 5 | **Stage 2 — EfficientNet-B3 (ONNX)** predicts emotion from the crop |
| 6 | API returns emotion, confidence, and `animalType` (`cat` or `dog`) |

### Key improvements
| Area | Detail |
|------|--------|
| **Pet validation** | Non-pet images are rejected before emotion analysis |
| **Pet type** | Response includes `animalType`: `cat` or `dog` |
| **Emotion model** | Self-hosted, trained on pet emotion datasets |
| **Training accuracy** | ~87.6% validation accuracy (4 classes: angry, calm, happy, sad) |
| **Speed** | ONNX-optimized inference for faster CPU performance |
| **Cost** | No per-image Nyckel fee for emotion (Nyckel kept as fallback only) |

### Audio scans
- Still uses AssemblyAI (unchanged for now)
- Pet-specific local audio classifier planned as a future upgrade

---

## ONNX Optimization (Emotion Model)

### What is ONNX?

**ONNX** (Open Neural Network Exchange) is an industry-standard format for running AI models efficiently on a server. After training the emotion model in PyTorch, we export it to ONNX so production inference is faster and uses less memory.

### Why we use it

| Benefit | What it means for PetMood |
|---------|---------------------------|
| **Faster scans** | Emotion prediction is ~30–40% quicker on CPU than raw PyTorch |
| **Lower memory** | Important on our Render server (2 GB RAM shared with YOLOv8 + Django) |
| **Same accuracy** | ONNX uses the same trained weights — no loss in prediction quality |
| **Production-ready** | ONNX Runtime is widely used in commercial AI deployments |

### How it fits in the pipeline

```
Training (offline) and Google Colab(Online):
  PyTorch model (.pth)  →  export  →  ONNX model (.onnx)

Production (every scan)
  Cropped pet image  →  ONNX Runtime  →  emotion + confidence
```

- **Stage 1 (YOLOv8):** still uses PyTorch/Ultralytics for pet detection
- **Stage 2 (emotion):** uses **ONNX** when `pet_emotion_b3.onnx` is available
- **Fallback:** if ONNX fails to load, the server automatically uses the PyTorch `.pth` file, then Nyckel API as a last resort

### Files on the server

| File | Purpose |
|------|---------|
| `pet_emotion_b3.onnx` | Primary — fast emotion inference (production) |
| `pet_emotion_b3.pth` | Backup — PyTorch weights |
| `pet_emotion_b3_classes.json` | Emotion labels: angry, calm, happy, sad |

When a scan succeeds, the API may return `"aiDetectorType": "efficientnet_b3_onnx"` to indicate the optimized model was used.

---

## Production Server Configuration (Complete)

| Item | Configuration |
|------|----------------|
| **Host** | Render (Standard plan — 2 GB RAM) |
| **Production URL** | https://ai-pet-mood.onrender.com |
| **Web server** | Gunicorn (production-grade, not dev runserver) |
| **Static files** | WhiteNoise + `collectstatic` (admin panel styling fixed) |
| **ML models** | YOLOv8-s + EfficientNet-B3 (ONNX) loaded at server startup |
| **Database** | SQLite + Firebase Firestore (unchanged) |
| **Storage** | Firebase Storage for scan media (unchanged) |

### Build command (Render)
```bash
pip install -r requirements.txt && python scripts/download_models.py && python manage.py collectstatic --noinput && python manage.py migrate && python manage.py createsuperuser --noinput || true
```

### Start command (Render)
```bash
gunicorn pet_emotion_detection.wsgi:application --bind 0.0.0.0:$PORT --workers 1 --timeout 120 --preload
```

---

## API Changes for Mobile App

### New behavior — no pet detected (image scans)

When the uploaded image does not contain a cat or dog:

**HTTP 422** response:
```json
{
  "error": "no_pet_in_image",
  "detail": "No cat or dog was detected in this photo. Please upload a clear photo of your pet.",
  "suggestions": [
    "Make sure your pet is clearly visible",
    "Ensure good lighting",
    "Get closer to your pet",
    "Avoid photos where the pet is very small"
  ]
}
```

The app should show this message to the user and ask them to retake the photo.

### Successful scan response (additional fields)

```json
{
  "emotion": "happy",
  "confidence": 0.91,
  "animalType": "dog",
  "analysisMethod": "two_stage_yolo_efficientnet",
  "aiDetectorType": "efficientnet_b3_onnx",
  "topEmotions": [...],
  "mediaUrl": "...",
  "petId": "..."
}
```

### Health / status endpoint
```
GET /api/ai-detector-status
```
Use this to verify the ML pipeline is loaded on the server.

---

## Links

| Resource | URL |
|----------|-----|
| API base | https://ai-pet-mood.onrender.com/api/ |
| Swagger docs | https://ai-pet-mood.onrender.com/swagger/ |
| Admin panel | https://ai-pet-mood.onrender.com/admin-panel/ |
| Health check | https://ai-pet-mood.onrender.com/api/health |

---

## Next Steps (Roadmap)

1. Monitor scan accuracy with real user uploads
2. Retrain emotion model with more data to improve beyond 87%
3. Upgrade audio scans with pet-specific local classifier
4. Optional: body posture detection (ears, tail) in a future release

---

*For technical questions, contact the backend team.*
