"""
Stage 1 — Pet Detection Gate
Uses YOLOv8 to verify a cat or dog exists before emotion analysis runs.
"""

import io
import logging
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# COCO class IDs: cat=15, dog=16
PET_CLASS_IDS = {15: "cat", 16: "dog"}
MIN_CONFIDENCE = 0.45
MIN_AREA_RATIO = 0.03
CROP_PADDING = 0.12


@dataclass
class DetectionResult:
    pet_detected: bool
    pet_type: Optional[str]
    detection_confidence: float
    bounding_box: Optional[Tuple]
    cropped_image_bytes: Optional[bytes]
    rejection_reason: Optional[str]


class PetDetector:

    def __init__(self):
        self._model = None
        self._ready = False
        self._load_model()

    def _load_model(self):
        model_size = os.getenv("YOLO_MODEL_SIZE", "s")
        model_name = f"yolov8{model_size}.pt"
        local_path = os.path.join(
            os.path.dirname(__file__), "models", model_name
        )
        try:
            from ultralytics import YOLO

            if os.path.exists(local_path):
                logger.info("Loading YOLOv8 from local: %s", local_path)
                self._model = YOLO(local_path)
            else:
                logger.info("Downloading YOLOv8-%s...", model_size)
                self._model = YOLO(model_name)
                import shutil

                default = os.path.expanduser(
                    f"~/.config/Ultralytics/{model_name}"
                )
                if os.path.exists(default):
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    shutil.copy(default, local_path)
                elif os.path.exists(model_name):
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    shutil.copy(model_name, local_path)
            self._ready = True
            logger.info("YOLOv8-%s ready", model_size)
        except ImportError:
            logger.error("ultralytics not installed")
        except Exception as e:
            logger.error("YOLOv8 load failed: %s", e)

    def detect(self, image_data: bytes) -> DetectionResult:
        if not self._ready or self._model is None:
            logger.warning("Detector not ready, failing open")
            return DetectionResult(
                pet_detected=True,
                pet_type=None,
                detection_confidence=0.0,
                bounding_box=None,
                cropped_image_bytes=image_data,
                rejection_reason=None,
            )

        try:
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            img_w, img_h = image.size
            img_array = np.array(image)

            results = self._model(
                img_array,
                verbose=False,
                conf=MIN_CONFIDENCE,
                classes=list(PET_CLASS_IDS.keys()),
            )[0]

            detections = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                if cls_id in PET_CLASS_IDS:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append({
                        "pet_type": PET_CLASS_IDS[cls_id],
                        "confidence": float(box.conf[0]),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "area_ratio": ((x2 - x1) * (y2 - y1)) / (img_w * img_h),
                    })

            if not detections:
                return DetectionResult(
                    pet_detected=False,
                    pet_type=None,
                    detection_confidence=0.0,
                    bounding_box=None,
                    cropped_image_bytes=None,
                    rejection_reason="no_pet_in_image",
                )

            best = max(detections, key=lambda d: d["confidence"])

            if best["area_ratio"] < MIN_AREA_RATIO:
                return DetectionResult(
                    pet_detected=False,
                    pet_type=best["pet_type"],
                    detection_confidence=best["confidence"],
                    bounding_box=None,
                    cropped_image_bytes=None,
                    rejection_reason="pet_too_small",
                )

            pad_x = (best["x2"] - best["x1"]) * CROP_PADDING
            pad_y = (best["y2"] - best["y1"]) * CROP_PADDING
            cx1 = max(0, int(best["x1"] - pad_x))
            cy1 = max(0, int(best["y1"] - pad_y))
            cx2 = min(img_w, int(best["x2"] + pad_x))
            cy2 = min(img_h, int(best["y2"] + pad_y))

            cropped = image.crop((cx1, cy1, cx2, cy2))
            buf = io.BytesIO()
            cropped.save(buf, format="JPEG", quality=92)

            return DetectionResult(
                pet_detected=True,
                pet_type=best["pet_type"],
                detection_confidence=round(best["confidence"], 4),
                bounding_box=(cx1 / img_w, cy1 / img_h, cx2 / img_w, cy2 / img_h),
                cropped_image_bytes=buf.getvalue(),
                rejection_reason=None,
            )

        except Exception as e:
            logger.error("Detection error: %s", e, exc_info=True)
            return DetectionResult(
                pet_detected=True,
                pet_type=None,
                detection_confidence=0.0,
                bounding_box=None,
                cropped_image_bytes=image_data,
                rejection_reason=None,
            )


_instance: Optional[PetDetector] = None


def get_pet_detector() -> PetDetector:
    global _instance
    if _instance is None:
        _instance = PetDetector()
    return _instance
