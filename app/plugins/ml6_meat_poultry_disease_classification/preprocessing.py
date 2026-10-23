"""Image preprocessing — transcribed from the delivered src/data_processing/preprocess.py.

The AI team uses the same transform for train, val and test (no augmentation):

- thermal: CLAHE on the L channel of Lab (clip 3.0, 8x8 tiles) → ToTensor → ImageNet Normalize,
  at the camera's native resolution (no resize).
- rgb: Resize((256, 256)) → ToTensor → ImageNet Normalize.

Images are decoded like torchvision's ImageFolder default loader:
``Image.open(...).convert("RGB")``.
"""
# cv2 is a C extension pylint cannot introspect (same false positive as ml4/ml2).
# pylint: disable=no-member
import io

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from app.domain.services.exceptions import InvalidImageError
from app.plugins.ml6_meat_poultry_disease_classification.constants import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID,
    IMAGENET_MEAN,
    IMAGENET_STD,
    RGB_RESIZE,
    TASK_THERMAL,
    THERMAL_NATIVE_SIZE,
)


def boost_contrast(image: Image.Image) -> np.ndarray:
    """CLAHE on the luminance channel only (preprocess.py::boost_contrast)."""
    array = np.array(image, dtype=np.uint8)
    lab = cv2.cvtColor(array, cv2.COLOR_RGB2LAB)
    lightness, chan_a, chan_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    merged = cv2.merge((clahe.apply(lightness), chan_a, chan_b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


_THERMAL_TRANSFORM = transforms.Compose([
    transforms.Lambda(boost_contrast),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

_RGB_TRANSFORM = transforms.Compose([
    transforms.Resize(RGB_RESIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def get_transform(task: str) -> transforms.Compose:
    """Return the (train = test) transform of an ensemble."""
    return _THERMAL_TRANSFORM if task == TASK_THERMAL else _RGB_TRANSFORM


def decode_image(image_bytes: bytes) -> Image.Image:
    """Decode raw bytes into an RGB PIL image, or raise InvalidImageError."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        return image.convert("RGB")
    except Exception as exc:
        raise InvalidImageError(f"No se puede decodificar la imagen: {exc}") from exc


def load_image(path: str) -> Image.Image:
    """Load an image file exactly like torchvision's ImageFolder default loader."""
    with open(path, "rb") as fh:
        return Image.open(fh).convert("RGB")


def to_tensor(image: Image.Image, task: str) -> torch.Tensor:
    """Apply the ensemble's transform and add the batch dimension: ``(1, 3, H, W)``."""
    return get_transform(task)(image).unsqueeze(0)


def input_warnings(image: Image.Image, task: str) -> list[str]:
    """Non-blocking warnings about inputs outside the training distribution."""
    if task == TASK_THERMAL and image.size != THERMAL_NATIVE_SIZE:
        width, height = THERMAL_NATIVE_SIZE
        return [
            f"La imagen térmica mide {image.size[0]}x{image.size[1]} y el ensemble se entrenó con "
            f"la resolución nativa de la cámara ({width}x{height}, ancho x alto, sin "
            "redimensionar): "
            "la predicción puede no ser fiable."
        ]
    return []
