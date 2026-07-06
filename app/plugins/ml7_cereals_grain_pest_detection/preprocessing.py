"""Image preprocessing for the grain pest-detection plugin."""
import base64
import io
import os

import numpy as np
from PIL import Image

from app.domain.services.exceptions import InvalidImageError
from app.plugins.ml7_cereals_grain_pest_detection.constants import IMAGE_EXTENSIONS


def image_path_to_numpy(image_path: str) -> np.ndarray:
    """Load an image from disk as an RGB numpy array."""
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in IMAGE_EXTENSIONS:
        raise InvalidImageError(f"Unsupported extension: {ext}. Supported: {IMAGE_EXTENSIONS}")
    try:
        return np.array(Image.open(image_path).convert("RGB"))
    except Exception as exc:
        raise InvalidImageError(f"Cannot open image at {image_path}: {exc}") from exc


def image_base64_to_numpy(b64: str) -> np.ndarray:
    """Decode a base64 JPEG/PNG into an RGB numpy array."""
    try:
        return np.array(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))
    except Exception as exc:
        raise InvalidImageError(f"Cannot decode base64 image: {exc}") from exc
