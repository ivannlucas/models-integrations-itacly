from __future__ import annotations

import base64
import io
import logging

import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def image_base64_to_tensor(image_b64: str) -> torch.Tensor:
    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return INFERENCE_TRANSFORM(image).unsqueeze(0)


def image_path_to_tensor(path: str) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return INFERENCE_TRANSFORM(image).unsqueeze(0)
