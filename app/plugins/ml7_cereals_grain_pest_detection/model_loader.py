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
    ultralytics' own default resolution (same pattern as
    ml10_dairy_disease_vector_detection/model_loader.py).
    """
    # Lazy: importing this module must not import torch (ultralytics/torch load on demand).
    from app.domain.services.torch_device import (  # noqa: PLC0415 # pylint: disable=import-outside-toplevel
        safe_torch_device,
    )

    return safe_torch_device()


def load_yolo():
    """Load the YOLO best.pt checkpoint (downloads from S3 if configured)."""
    from ultralytics import YOLO  # noqa: PLC0415 # pylint: disable=import-outside-toplevel

    model_path = _store.path(MODEL_FILENAME)
    model = YOLO(str(model_path))
    logger.info("Ml7CerealsGrainPestDetection YOLO loaded from %s", model_path)
    return model
