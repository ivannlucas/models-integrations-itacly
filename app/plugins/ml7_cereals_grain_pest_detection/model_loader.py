"""Loads the YOLO detector from ArtifactStore. ultralytics is imported lazily."""
import logging

from app.infrastructure.artifact_store import ArtifactStore
from app.plugins.ml7_cereals_grain_pest_detection.constants import (
    ARTIFACT_FOLDER_NAME,
    MODEL_FILENAME,
)

logger = logging.getLogger(__name__)

_store = ArtifactStore(ARTIFACT_FOLDER_NAME)


def load_yolo():
    """Load the YOLO best.pt checkpoint (downloads from S3 if configured)."""
    from ultralytics import YOLO  # noqa: PLC0415 — heavy import kept lazy

    model_path = _store.path(MODEL_FILENAME)
    model = YOLO(str(model_path))
    logger.info("Ml7CerealsGrainPestDetection YOLO loaded from %s", model_path)
    return model
