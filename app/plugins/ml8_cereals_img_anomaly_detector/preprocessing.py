import base64
import io
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from app.domain.services.exceptions import InvalidImageError
from app.plugins.ml8_cereals_img_anomaly_detector.constants import IMAGE_SIZE

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


_DEFAULT_TRANSFORM = _build_transform(IMAGE_SIZE)


def image_base64_to_tensor(
    image_base64: str,
    image_size: int = IMAGE_SIZE,
) -> torch.Tensor:
    """Decode a base64-encoded image and return a (1, C, H, W) float tensor."""
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
    """Load an image from disk and return a (1, C, H, W) float tensor."""
    return image_path_to_tensor_and_image(image_path, image_size=image_size)[0]


def image_path_to_tensor_and_image(
    image_path: str | Path,
    image_size: int = IMAGE_SIZE,
) -> tuple[torch.Tensor, Image.Image]:
    """Load an image from disk and return both its (1, C, H, W) tensor and the decoded
    PIL image — batch prediction reuses the same decoded image to also produce the
    annotated_image thumbnail, instead of opening the file from disk twice."""
    try:
        image = Image.open(str(image_path)).convert("RGB")
    except Exception as exc:
        raise InvalidImageError(f"Cannot open image at {image_path}: {exc}") from exc

    transform = _DEFAULT_TRANSFORM if image_size == IMAGE_SIZE else _build_transform(image_size)
    return transform(image).unsqueeze(0), image
