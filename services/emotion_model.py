"""
Shared emotion model architecture, preprocessing, and result parsing.

Used by:
- services/pet_emotion_classifier.py (production inference)
- scripts/train.py (training)
- scripts/export_onnx.py (ONNX conversion)
"""

from __future__ import annotations

import io
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

EMOTION_CLASSES = [
    "happy",
    "sad",
    "angry",
    "calm",
    "anxious",
    "excited",
    "fearful",
    "playful",
]

CONFIDENCE_THRESHOLDS = {
    "happy": 0.35,
    "calm": 0.35,
    "excited": 0.38,
    "playful": 0.38,
    "sad": 0.40,
    "anxious": 0.42,
    "angry": 0.45,
    "fearful": 0.45,
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_efficientnet_b3(num_classes: int | None = None):
    """Build EfficientNet-B3 with the custom emotion classification head."""
    import torch.nn as nn
    from torchvision import models

    num_classes = num_classes or len(EMOTION_CLASSES)
    backbone = models.efficientnet_b3(
        weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
    )
    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=0.35, inplace=True),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes),
    )
    return backbone


def _center_crop(image: Image.Image, size: int) -> Image.Image:
    w, h = image.size
    if w < size or h < size:
        return image.resize((size, size), Image.BICUBIC)
    left = (w - size) // 2
    top = (h - size) // 2
    return image.crop((left, top, left + size, top + size))


def preprocess_pil_image(image: Image.Image) -> np.ndarray:
    """
    Resize → center-crop 224 → normalize → NCHW float32 batch.
    Matches torchvision transforms used during training.
    """
    image = image.convert("RGB")
    image = image.resize((256, 256), Image.BICUBIC)
    image = _center_crop(image, 224)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0)


def preprocess_image_bytes(image_data: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_data))
    return preprocess_pil_image(image)


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64)
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return (exp / np.sum(exp)).astype(np.float32)


def classes_json_path(weights_path: str) -> str:
    return weights_path.replace(".pth", "_classes.json").replace(".onnx", "_classes.json")


def load_model_classes(weights_path: str) -> List[str]:
    """Load trained class names from sidecar JSON; fallback to default 8 classes."""
    mapping_path = classes_json_path(weights_path)
    if os.path.exists(mapping_path):
        with open(mapping_path, encoding="utf-8") as f:
            data = json.load(f)
        classes = data.get("classes")
        if isinstance(classes, list) and classes:
            return classes
    return list(EMOTION_CLASSES)


def parse_emotion_probs(
    probs: np.ndarray,
    pet_type: Optional[str] = None,
    *,
    classes: Optional[List[str]] = None,
    finetuned: bool = False,
    backend: str = "efficientnet_b3",
) -> Dict[str, Any]:
    """Turn raw softmax probabilities into the API emotion result dict."""
    emotion_classes = classes or EMOTION_CLASSES
    sorted_idx = np.argsort(probs)[::-1]
    top_emotion = emotion_classes[sorted_idx[0]]
    top_conf = float(probs[sorted_idx[0]])

    threshold = CONFIDENCE_THRESHOLDS.get(top_emotion, 0.38)
    if top_conf < threshold:
        for idx in sorted_idx[1:3]:
            alt = emotion_classes[idx]
            alt_conf = float(probs[idx])
            if alt_conf >= CONFIDENCE_THRESHOLDS.get(alt, 0.38):
                top_emotion = alt
                top_conf = alt_conf
                break
        else:
            top_emotion = "calm"
            calm_idx = emotion_classes.index("calm") if "calm" in emotion_classes else sorted_idx[0]
            top_conf = float(probs[calm_idx])

    top_5: List[Dict] = [
        {
            "emotion": emotion_classes[i],
            "confidence": round(float(probs[i]), 4),
        }
        for i in sorted_idx[: min(5, len(emotion_classes))]
    ]

    if finetuned:
        detector_type = f"{backend}_finetuned"
    else:
        detector_type = f"{backend}_pretrained"

    return {
        "emotion": top_emotion,
        "confidence": round(top_conf, 4),
        "top_emotions": top_5,
        "ai_detector_type": detector_type,
        "analysis_method": "two_stage_yolo_efficientnet",
        "detected_pet_type": pet_type,
    }


def default_model_paths() -> Tuple[str, str]:
    base = os.path.join(os.path.dirname(__file__), "models")
    return (
        os.getenv("PET_EMOTION_MODEL_PATH", os.path.join(base, "pet_emotion_b3.pth")),
        os.getenv("PET_EMOTION_ONNX_PATH", os.path.join(base, "pet_emotion_b3.onnx")),
    )
