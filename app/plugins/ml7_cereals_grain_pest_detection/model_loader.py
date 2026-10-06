"""Loads the YOLO detector from ArtifactStore. ultralytics is imported lazily."""
import logging

from app.infrastructure.artifact_store import ArtifactStore
from app.plugins.ml7_cereals_grain_pest_detection.constants import (
    ARTIFACT_FOLDER_NAME,
    MODEL_FILENAME,
)

logger = logging.getLogger(__name__)

_store = ArtifactStore(ARTIFACT_FOLDER_NAME)


def safe_device():
    """Return CUDA only if it can actually execute network operations; otherwise CPU.

    ``torch.cuda.is_available()`` returns True for GPUs whose compute capability is
    unsupported by the installed PyTorch build (e.g. sm_61), but real ops then fail.
    Ultralytics does its own independent ``torch.cuda.is_available()`` check when
    ``device=`` is not passed explicitly, so every ``model.predict()``/``model.train()``
    call in this plugin must pass this resolved device explicitly — never rely on
    ultralytics' own default resolution (same pattern as ml10_dairy_disease_vector_detection/model_loader.py).
    """
    import torch  # noqa: PLC0415 — heavy import kept lazy

    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        torch.nn.Conv2d(1, 1, 1).cuda()(torch.zeros(1, 1, 4, 4).cuda())
        return torch.device("cuda")
    except Exception:  # pylint: disable=broad-exception-caught
        logger.warning("CUDA detectada pero no funcional para operaciones de red — usando CPU.")
        return torch.device("cpu")


def load_yolo():
    """Load the YOLO best.pt checkpoint (downloads from S3 if configured)."""
    from ultralytics import YOLO  # noqa: PLC0415 — heavy import kept lazy

    model_path = _store.path(MODEL_FILENAME)
    model = YOLO(str(model_path))
    logger.info("Ml7CerealsGrainPestDetection YOLO loaded from %s", model_path)
    return model
