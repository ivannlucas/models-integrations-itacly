import logging

import torch
import torch.nn as nn
from torchvision import models

from app.infrastructure.artifact_store import ArtifactStore
from app.plugins.ml8_cereals_img_anomaly_detector.constants import (
    ARTIFACT_FOLDER_NAME,
    CATEGORY_NAMES,
    CEREAL_NAMES,
    IMAGE_SIZE,
    MODEL_FILENAME,
)

logger = logging.getLogger(__name__)

_store = ArtifactStore(ARTIFACT_FOLDER_NAME)


class _MultiTaskMobileNetV3Large(nn.Module):
    def __init__(self, num_classes: int, num_cereals: int) -> None:
        super().__init__()
        base = models.mobilenet_v3_large(weights=None)
        self.features = base.features
        self.avgpool = base.avgpool
        self.neck = nn.Sequential(
            base.classifier[0],
            base.classifier[1],
            base.classifier[2],
        )
        num_features: int = base.classifier[3].in_features
        self.classifier_categoria = nn.Linear(num_features, num_classes)
        self.classifier_cereal = nn.Linear(num_features, num_cereals)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.neck(x)
        return self.classifier_categoria(x), self.classifier_cereal(x)


class _MultiTaskEfficientNetB0(nn.Module):
    def __init__(self, num_classes: int, num_cereals: int) -> None:
        super().__init__()
        base = models.efficientnet_b0(weights=None)
        self.features = base.features
        self.avgpool = base.avgpool
        self.dropout = base.classifier[0]
        num_features: int = base.classifier[1].in_features
        self.classifier_categoria = nn.Linear(num_features, num_classes)
        self.classifier_cereal = nn.Linear(num_features, num_cereals)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier_categoria(x), self.classifier_cereal(x)


class _MultiTaskResNet18(nn.Module):
    def __init__(self, num_classes: int, num_cereals: int) -> None:
        super().__init__()
        base = models.resnet18(weights=None)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.dropout = nn.Dropout(p=0.2)
        num_features: int = base.fc.in_features
        self.classifier_categoria = nn.Linear(num_features, num_classes)
        self.classifier_cereal = nn.Linear(num_features, num_cereals)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier_categoria(x), self.classifier_cereal(x)


_ARCH_CLASSES = {
    "mobilenet_v3_large": _MultiTaskMobileNetV3Large,
    "efficientnet_b0": _MultiTaskEfficientNetB0,
    "resnet18": _MultiTaskResNet18,
}


def load_model_bundle() -> dict:
    """Load the PyTorch model from the checkpoint stored in ArtifactStore."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = _store.path(MODEL_FILENAME)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    arch: str = checkpoint.get("model_name", "mobilenet_v3_large")
    num_classes: int = checkpoint.get("num_classes", len(CATEGORY_NAMES))
    num_cereals: int = checkpoint.get("num_cereales", len(CEREAL_NAMES))
    idx_to_class: dict = checkpoint.get("idx_to_class", {i: n for i, n in enumerate(CATEGORY_NAMES)})
    idx_to_cereal: dict = checkpoint.get("idx_to_cereal", {i: n for i, n in enumerate(CEREAL_NAMES)})

    model_cls = _ARCH_CLASSES.get(arch)
    if model_cls is None:
        raise ValueError(f"Unsupported architecture '{arch}'. Supported: {list(_ARCH_CLASSES)}")

    model = model_cls(num_classes, num_cereals)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(
        "Ml8CerealsImgAnomalyDetectorPlugin bundle ready — arch=%s device=%s",
        arch, device,
    )

    return {
        "model_id": "ml8-cereals-img-anomaly-detector",
        "arch": arch,
        "model": model,
        "device": device,
        "image_size": IMAGE_SIZE,
        "idx_to_class": idx_to_class,
        "idx_to_cereal": idx_to_cereal,
    }
