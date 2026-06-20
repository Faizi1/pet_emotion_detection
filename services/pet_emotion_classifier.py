"""
Stage 2 — Emotion Classification
EfficientNet-B3 fine-tuned on pet emotion datasets.

Inference priority:
  1. ONNX Runtime (fastest on CPU) — if .onnx file exists
  2. PyTorch — if .pth file exists
  3. Nyckel API fallback
"""

import logging
import os
from typing import Any, Dict, Optional

import numpy as np

from .emotion_model import (
    build_efficientnet_b3,
    default_model_paths,
    load_model_classes,
    parse_emotion_probs,
    preprocess_image_bytes,
    softmax,
)

logger = logging.getLogger(__name__)


def _use_onnx_preference() -> bool:
    value = os.getenv("USE_ONNX_INFERENCE", "auto").strip().lower()
    if value in ("0", "false", "no"):
        return False
    if value in ("1", "true", "yes"):
        return True
    # auto: use ONNX when the file exists
    _, onnx_path = default_model_paths()
    return os.path.exists(onnx_path)


class PetEmotionClassifier:

    def __init__(self):
        self._torch_model = None
        self._onnx_session = None
        self._ready = False
        self._finetuned = False
        self._backend = "efficientnet_b3"
        self._device = None
        self._pth_path, self._onnx_path = default_model_paths()
        self._classes: list[str] = []
        self._load()

    def _load(self):
        try:
            if _use_onnx_preference() and os.path.exists(self._onnx_path):
                self._load_onnx()
                if self._ready:
                    return

            self._load_pytorch()
        except ImportError as e:
            logger.error("ML dependencies not installed: %s", e)
        except Exception as e:
            logger.error("Classifier load failed: %s", e)

    def _load_onnx(self):
        import onnxruntime as ort

        self._classes = load_model_classes(self._onnx_path)
        self._onnx_session = ort.InferenceSession(
            self._onnx_path,
            providers=["CPUExecutionProvider"],
        )
        self._ready = True
        self._finetuned = True
        self._backend = "efficientnet_b3_onnx"
        logger.info(
            "ONNX emotion model loaded from %s (%d classes)",
            self._onnx_path,
            len(self._classes),
        )

    def _load_pytorch(self):
        import torch

        self._device = torch.device("cpu")
        weights_path = self._pth_path if os.path.exists(self._pth_path) else ""
        self._classes = load_model_classes(weights_path) if weights_path else []
        num_classes = len(self._classes) if self._classes else 8

        self._torch_model = build_efficientnet_b3(num_classes=num_classes).to(self._device)
        self._torch_model.eval()
        self._ready = True
        logger.info("EfficientNet-B3 PyTorch model built")

        if weights_path:
            state = torch.load(weights_path, map_location=self._device, weights_only=True)
            self._torch_model.load_state_dict(state)
            self._finetuned = True
            logger.info(
                "Fine-tuned PyTorch weights loaded from %s (%d classes: %s)",
                weights_path,
                len(self._classes),
                self._classes,
            )
        else:
            logger.warning(
                "No weights at %s. Using ImageNet pretrained head.",
                self._pth_path,
            )

    def classify(
        self,
        image_data: bytes,
        pet_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self._ready:
            return self._nyckel_fallback(image_data, pet_type)

        try:
            batch = preprocess_image_bytes(image_data)

            if self._onnx_session is not None:
                logits = self._onnx_session.run(
                    None,
                    {"input": batch.astype(np.float32)},
                )[0][0]
                probs = softmax(logits)
                backend = "efficientnet_b3_onnx"
            else:
                import torch

                tensor = torch.from_numpy(batch).to(self._device)
                with torch.no_grad():
                    logits = self._torch_model(tensor)
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                backend = "efficientnet_b3"

            result = parse_emotion_probs(
                probs,
                pet_type=pet_type,
                classes=self._classes or None,
                finetuned=self._finetuned,
                backend=backend,
            )
            return result

        except Exception as e:
            logger.error("Classification error: %s", e, exc_info=True)
            return self._nyckel_fallback(image_data, pet_type)

    def _nyckel_fallback(
        self,
        image_data: bytes,
        pet_type: Optional[str],
    ) -> Dict[str, Any]:
        try:
            from .nyckel_service import nyckel_service

            result = nyckel_service.detect_emotion_from_image(image_data)
            result["analysis_method"] = "nyckel_fallback"
            result["detected_pet_type"] = pet_type
            return result
        except Exception as e:
            logger.error("Nyckel fallback failed: %s", e)
            return {
                "emotion": "calm",
                "confidence": 0.5,
                "top_emotions": [{"emotion": "calm", "confidence": 0.5}],
                "ai_detector_type": "fallback",
                "analysis_method": "error_fallback",
                "detected_pet_type": pet_type,
            }


_classifier_instance: Optional[PetEmotionClassifier] = None


def get_emotion_classifier() -> PetEmotionClassifier:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = PetEmotionClassifier()
    return _classifier_instance
