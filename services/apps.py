from django.apps import AppConfig
import logging

logger = logging.getLogger(__name__)


class ServicesConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "services"

    def ready(self):
        import os

        if os.environ.get("DJANGO_SKIP_MODEL_WARMUP"):
            return

        try:
            logger.info("Warming up ML models...")
            from .pet_detector import get_pet_detector

            detector = get_pet_detector()
            logger.info("YOLOv8 ready: %s", detector._ready)

            from .pet_emotion_classifier import get_emotion_classifier

            classifier = get_emotion_classifier()
            logger.info(
                "Emotion classifier ready: %s | fine-tuned: %s | backend: %s",
                classifier._ready,
                classifier._finetuned,
                classifier._backend,
            )
        except Exception as e:
            logger.error("Model warmup error: %s", e)
