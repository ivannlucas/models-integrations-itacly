"""Funciones de preprocesamiento para el plugin Ml10DairyDiseaseVectorDetection."""
from __future__ import annotations

import base64
import io
import logging

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from app.domain.services.exceptions import InvalidImageError

logger = logging.getLogger(__name__)


def build_eval_transform(imgsz: int = 224) -> transforms.Compose:
    """Preprocessing transform applied to a detector crop before classification (production).

    Must stay identical to ``predict_full_pipeline`` in the original delivered
    ``src/predict/predictor.py`` (also mirrored in ``notebooks/evaluation/pipeline.ipynb``) —
    the only two places in the delivered code that actually run the real two-stage
    detect-crop-classify flow on a raw image, i.e. the production path this plugin's
    ``predict_inline``/``predict_batch`` replicate. Both use ``Resize(256)+CenterCrop(224)``,
    NOT the plain square ``Resize((224, 224))`` used by ``trainer.py``/``predict_classifier``
    to train/evaluate the classifier in isolation on already-cropped-to-224 ImageFolder images
    (a different, narrower evaluation that never exercises the detector or the raw-image crop
    step — see ``_cls_transforms`` in plugin.py, which intentionally does NOT reuse this
    function for that reason).

    A prior version of this function used the plain-resize variant here too, reasoning that it
    "matched training" — that reasoning was wrong: it matched the classifier's OWN train/eval
    recipe, not the full pipeline's real inference preprocessing. Verified empirically against
    the one real delivered end-to-end example (``inbox/a10/codigo/.../data/test.jpg`` +
    ``data/predictions/test_results.json``, produced by the original ``predict_full_pipeline``
    against the exact artifacts vendored in this plugin): the tick crop at bbox
    (755,280,840,374) scores cls_conf=0.9996 with ``Resize(256)+CenterCrop(224)`` — matching
    the delivered JSON to 4 decimals — vs. 0.9991 with the plain-resize variant. Small in this
    case, but the two transforms are NOT equivalent and only one matches the delivered
    production code and its own documented golden output.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(imgsz),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# Transformación real aplicada a cada crop del detector antes de clasificar.
CLASSIFIER_TRANSFORM = build_eval_transform(224)


def image_base64_to_pil(image_b64: str) -> Image.Image:
    """Decodifica base64 → imagen PIL RGB."""
    try:
        image_bytes = base64.b64decode(image_b64)
    except (base64.binascii.Error, ValueError, OSError) as exc:
        logger.error("Invalid base64 input: %s", exc)
        raise InvalidImageError(f"Invalid base64 image data: {exc}") from exc
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        logger.error("Failed to decode image from base64: %s", exc)
        raise InvalidImageError(f"Failed to decode image from base64: {exc}") from exc


def image_path_to_pil(path: str) -> Image.Image:
    """Carga imagen desde disco → PIL RGB."""
    try:
        return Image.open(path).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        logger.error("Failed to load image from path %s: %s", path, exc)
        raise InvalidImageError(f"Failed to load image from {path}: {exc}") from exc


def crop_to_tensor(image_pil: Image.Image, x1: int, y1: int, x2: int, y2: int) -> torch.Tensor:
    """Recorta una región de la imagen PIL y la convierte en tensor (1, 3, 224, 224)."""
    try:
        crop = image_pil.crop((x1, y1, x2, y2))
        return CLASSIFIER_TRANSFORM(crop).unsqueeze(0)
    except (OSError, ValueError) as exc:
        logger.error("Failed to crop image region (%d,%d,%d,%d): %s", x1, y1, x2, y2, exc)
        raise InvalidImageError(f"Failed to crop image: {exc}") from exc
