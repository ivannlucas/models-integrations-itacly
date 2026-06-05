"""Funciones de preprocesamiento para el plugin Modelo10Lacteo."""
from __future__ import annotations

import base64
import io
import logging

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from app.domain.services.exceptions import InvalidImageError

logger = logging.getLogger(__name__)

# Transformación estándar ImageNet para el clasificador (igual que en entrenamiento)
CLASSIFIER_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


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
