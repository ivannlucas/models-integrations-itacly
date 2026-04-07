import io

import albumentations as A  # type: ignore[import]
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2  # type: ignore[import]
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
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise InvalidImageError(f"Could not decode image: {exc}") from exc

    img_array = np.array(image)
    transformed = _INFERENCE_TRANSFORM(image=img_array)
    tensor = transformed["image"]
    return tensor.unsqueeze(0)
