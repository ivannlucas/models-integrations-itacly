"""Model loading for the fungal leaf-disease CNN plugin.

Defines the ``LeafCNN`` architecture (which must match the training repository
exactly) and resolves/loads the ``.pth`` checkpoint via :class:`ArtifactStore`.
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.infrastructure.artifact_store import ArtifactStore
from app.plugins.ml2_fungal_cnn_disease_detection.constants import (
    ARTIFACT_FOLDER_NAME,
    CLASS_NAMES,
    IMAGE_SIZE,
    MODEL_FILENAME,
    MODEL_ID,
)

logger = logging.getLogger(__name__)

_store = ArtifactStore(ARTIFACT_FOLDER_NAME)


def _safe_device() -> torch.device:
    """Return a CUDA device only if it can actually execute model operations."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        torch.nn.Conv2d(1, 1, 1)(torch.zeros(1, 1, 4, 4).cuda())
        return torch.device("cuda")
    except Exception:
        logger.warning("CUDA detectada pero no funcional para operaciones de red — usando CPU.")
        return torch.device("cpu")


class LeafCNN(nn.Module):
    """CNN personalizada para clasificación de enfermedades en hojas de vid.

    Debe coincidir exactamente con la arquitectura usada en el entrenamiento.
    """

    def __init__(self, num_classes: int) -> None:
        """Build the convolutional feature extractor and the linear classifier."""
        super().__init__()

        self.features = nn.Sequential(
            # Bloque 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 224 → 112
            # Bloque 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112 → 56
            # Bloque 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """Run the forward pass and return per-class logits of shape ``(B, num_classes)``.

        When ``return_features`` is True, also returns the last conv block's raw feature
        map ``(B, 128, H, W)`` (before pooling) so a Class Activation Map can be computed
        from it — the classifier being a single Linear layer applied right after global
        average pooling makes CAM exact here (Grad-CAM reduces to CAM for this architecture).
        """
        feat = self.features(x)
        pooled = F.adaptive_avg_pool2d(feat, 1)  # Global Average Pooling → (B, 128, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (B, 128)
        logits = self.classifier(pooled)
        if return_features:
            return logits, feat
        return logits


def load_model_bundle() -> dict:
    """Load the LeafCNN checkpoint stored in :class:`ArtifactStore` and return a bundle dict."""
    device = _safe_device()
    model_path = _store.path(MODEL_FILENAME)

    model = LeafCNN(num_classes=len(CLASS_NAMES)).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    logger.info("Ml2FungalCnnDiseaseDetectionPlugin bundle ready — device=%s", device)

    return {
        "model_id": MODEL_ID,
        "model": model,
        "device": device,
        "image_size": IMAGE_SIZE,
        "classes": list(CLASS_NAMES),
    }
