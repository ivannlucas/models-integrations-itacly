"""Image preprocessing for the thermal-mastitis CNN (mirrors training transforms)."""
import io

import albumentations as A
import cv2
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

# Transcribed verbatim from the delivered src/data/transforms.py::get_train_transforms — used
# only by plugin.train() to fine-tune on user-supplied images. Inference always uses
# _INFERENCE_TRANSFORM above (no augmentation), matching get_val_transforms/
# get_inference_transforms.
_TRAIN_TRANSFORM = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.Affine(
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        scale=(0.8, 1.2),
        rotate=(-15, 15),
        border_mode=cv2.BORDER_CONSTANT,
        p=0.5,
    ),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(std_range=(0.02, 0.05), p=0.3),
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


def load_and_augment_for_training(image_path: str) -> torch.Tensor:
    """Load an image from disk and apply the training augmentation pipeline (unbatched)."""
    image = Image.open(image_path).convert("RGB")
    return _TRAIN_TRANSFORM(image=np.array(image))["image"]


def load_and_transform_for_eval(image_path: str) -> torch.Tensor:
    """Load an image from disk and apply the inference/validation transform (unbatched)."""
    image = Image.open(image_path).convert("RGB")
    return _INFERENCE_TRANSFORM(image=np.array(image))["image"]
