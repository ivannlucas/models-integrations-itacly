"""Image preprocessing for the fungal leaf-disease CNN plugin.

Mirrors the validation/inference transform of the training repository:
resize to 224×224, convert to tensor and normalize with mean/std 0.5.
"""
import base64
import io
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from app.domain.services.exceptions import InvalidImageError
from app.plugins.ml2_fungal_cnn_disease_detection.constants import IMAGE_SIZE

_NORM_MEAN = (0.5, 0.5, 0.5)
_NORM_STD = (0.5, 0.5, 0.5)


def _build_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Return the inference transform pipeline for the given square image size."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_NORM_MEAN, std=_NORM_STD),
    ])


_DEFAULT_TRANSFORM = _build_transform(IMAGE_SIZE)


def image_base64_to_tensor(
    image_base64: str,
    image_size: int = IMAGE_SIZE,
) -> torch.Tensor:
    """Decode a base64-encoded image and return a ``(1, C, H, W)`` float tensor."""
    try:
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise InvalidImageError(f"Cannot decode image from base64: {exc}") from exc

    transform = _DEFAULT_TRANSFORM if image_size == IMAGE_SIZE else _build_transform(image_size)
    return transform(image).unsqueeze(0)


def image_path_to_tensor(
    image_path: str | Path,
    image_size: int = IMAGE_SIZE,
) -> torch.Tensor:
    """Load an image from disk and return a ``(1, C, H, W)`` float tensor."""
    return image_path_to_tensor_and_image(image_path, image_size=image_size)[0]


def image_path_to_tensor_and_image(
    image_path: str | Path,
    image_size: int = IMAGE_SIZE,
) -> tuple[torch.Tensor, Image.Image]:
    """Load an image from disk and return both its ``(1, C, H, W)`` tensor and the decoded
    PIL image — batch prediction reuses the same decoded image to overlay the per-row CAM
    heatmap, instead of opening the file from disk twice."""
    try:
        image = Image.open(str(image_path)).convert("RGB")
    except Exception as exc:
        raise InvalidImageError(f"Cannot open image at {image_path}: {exc}") from exc

    transform = _DEFAULT_TRANSFORM if image_size == IMAGE_SIZE else _build_transform(image_size)
    return transform(image).unsqueeze(0), image
