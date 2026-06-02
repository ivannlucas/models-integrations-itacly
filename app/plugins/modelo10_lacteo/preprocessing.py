from __future__ import annotations

import base64
import io

import torch
from PIL import Image
from torchvision import transforms

# Transformación estándar ImageNet para el clasificador (igual que en entrenamiento)
CLASSIFIER_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def image_base64_to_pil(image_b64: str) -> Image.Image:
    """Decodifica base64 → imagen PIL RGB."""
    image_bytes = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def image_path_to_pil(path: str) -> Image.Image:
    """Carga imagen desde disco → PIL RGB."""
    return Image.open(path).convert("RGB")


def crop_to_tensor(image_pil: Image.Image, x1: int, y1: int, x2: int, y2: int) -> torch.Tensor:
    """Recorta una región de la imagen PIL y la convierte en tensor (1, 3, 224, 224)."""
    crop = image_pil.crop((x1, y1, x2, y2))
    return CLASSIFIER_TRANSFORM(crop).unsqueeze(0)
