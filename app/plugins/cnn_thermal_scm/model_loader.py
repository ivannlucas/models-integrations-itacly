import logging
from typing import Any

import torch
import torch.nn as nn
import timm  # type: ignore[import]

from app.infrastructure.artifact_store import ArtifactStore

logger = logging.getLogger(__name__)

_store = ArtifactStore("cnn_thermal_scm")

ARTIFACT_FILENAME = "baseline_efficientnet_final_model.pth"
BACKBONE = "efficientnet_b0"
NUM_CLASSES = 2
DROPOUT = 0.3


class BaselineModel(nn.Module):
    def __init__(
        self,
        backbone: str = BACKBONE,
        pretrained: bool = False,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="")
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feats = self.backbone(dummy)
            feature_dim = feats.shape[1] if feats.ndim == 4 else feats.shape[-1]
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(feature_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.ndim == 4:
            features = self.global_pool(features)
        features = features.flatten(1)
        return self.classifier(features)


def load_model() -> tuple[nn.Module, torch.device]:
    artifact_path = _store.path(ARTIFACT_FILENAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading model on device: %s", device)

    model = BaselineModel(backbone=BACKBONE, pretrained=False, num_classes=NUM_CLASSES, dropout=DROPOUT)
    checkpoint = torch.load(artifact_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    logger.info("Model loaded from %s", artifact_path)
    return model, device
