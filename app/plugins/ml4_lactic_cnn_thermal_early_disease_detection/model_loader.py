"""Loads the thermal-mastitis EfficientNet (timm) checkpoint via ArtifactStore."""
import logging

import timm
import torch
from torch import nn

from app.infrastructure.artifact_store import ArtifactStore
from app.plugins.ml4_lactic_cnn_thermal_early_disease_detection.constants import (
    ARTIFACT_FOLDER_NAME,
    BACKBONE,
    DROPOUT,
    MODEL_FILENAME,
    NUM_CLASSES,
)

logger = logging.getLogger(__name__)

_store = ArtifactStore(ARTIFACT_FOLDER_NAME)


def _safe_device() -> torch.device:
    """Return CUDA only if it can run a conv op; otherwise CPU (handles unsupported GPUs)."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        torch.nn.Conv2d(1, 1, 1)(torch.zeros(1, 1, 4, 4).cuda())
        return torch.device("cuda")
    except Exception:
        logger.warning("CUDA detectada pero no funcional — usando CPU.")
        return torch.device("cpu")


class BaselineModel(nn.Module):
    """EfficientNet-B0 backbone with a 2-class head (matches the training repo)."""

    def __init__(
        self,
        backbone: str = BACKBONE,
        pretrained: bool = False,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT,
    ) -> None:
        """Build the timm backbone and the dropout + linear classification head."""
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, global_pool="",
        )
        with torch.no_grad():
            feats = self.backbone(torch.randn(1, 3, 224, 224))
            feature_dim = feats.shape[1] if feats.ndim == 4 else feats.shape[-1]
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(feature_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the backbone, global pool and classifier; return per-class logits."""
        features = self.backbone(x)
        if features.ndim == 4:
            features = self.global_pool(features)
        return self.classifier(features.flatten(1))


def load_model() -> tuple[nn.Module, torch.device]:
    """Instantiate BaselineModel and load its state dict (plain or full checkpoint)."""
    device = _safe_device()
    model = BaselineModel(
        backbone=BACKBONE, pretrained=False, num_classes=NUM_CLASSES, dropout=DROPOUT,
    )
    checkpoint = torch.load(_store.path(MODEL_FILENAME), map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    logger.info("Ml4LacticCnnThermal model loaded (device=%s)", device)
    return model, device
