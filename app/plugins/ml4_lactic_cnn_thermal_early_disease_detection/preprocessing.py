"""Image preprocessing for the thermal-mastitis CNN (mirrors training transforms)."""
import io

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

from app.domain.services.exceptions import InvalidImageError

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_INFERENCE_TRANSFORM = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ToTensorV2(),
])


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Decode raw image bytes into a batched ``(1, 3, 224, 224)`` float tensor."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise InvalidImageError(
            f"Could not decode image: {exc}. Supported formats: JPEG, PNG, BMP."
        ) from exc

    tensor = _INFERENCE_TRANSFORM(image=np.array(image))["image"]
    return tensor.unsqueeze(0)
