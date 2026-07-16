# Frontend — Base URL Update

## Action required

Update the API base URL in the mobile app / frontend:

| | URL |
|---|-----|
| **Old** | `https://pet-emotion-detection.onrender.com` |
| **New** | `https://ai-pet-mood.onrender.com` |

### Example
```javascript
// Before
const BASE_URL = "https://pet-emotion-detection.onrender.com/api";

// After
const BASE_URL = "https://ai-pet-mood.onrender.com/api";
```

All API paths stay the same — only the host changes.  
Example: scan endpoint is still `POST /api/scans/` (confirm exact path in Swagger).

**Swagger:** https://ai-pet-mood.onrender.com/swagger/

---

## Image scan — handle new error (422)

If the user uploads a photo **without a pet**, the API now returns **HTTP 422** instead of a fake emotion.

Show the `detail` message from the response and prompt the user to retake the photo.

```json
{
  "error": "no_pet_in_image",
  "detail": "No cat or dog was detected in this photo. Please upload a clear photo of your pet.",
  "suggestions": ["..."]
}
```

---

## New response fields (optional UI)

On successful image scans you may receive:

- `animalType` — `"cat"` or `"dog"` (was `"unknown"` before)
- `analysisMethod` — e.g. `"two_stage_yolo_efficientnet"`
- `aiDetectorType` — e.g. `"efficientnet_b3_onnx"`

No breaking changes to existing fields (`emotion`, `confidence`, `mediaUrl`, `petId`).

---

## Quick test

1. Change base URL to `https://ai-pet-mood.onrender.com/api`
2. Upload a **pet photo** → should return emotion + `animalType`
3. Upload a **non-pet photo** → should return **422** with error message
