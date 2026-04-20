from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.infrastructure.artifact_store import ArtifactStore

logger = logging.getLogger(__name__)

_store = ArtifactStore("cnn_fungal_detection")

ARTIFACT_FILENAME = "leafcnn_best.pth"
CLASSES = ["black_rot", "downy_mildew", "healthy", "powdery_mildew", "trunk_disease"]
NUM_CLASSES = len(CLASSES)
ARTIFACT_PATH = _store.path(ARTIFACT_FILENAME)

class LeafCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def load_leafcnn(device: torch.device) -> LeafCNN:
    artifact_path = _store.path(ARTIFACT_FILENAME)
    model = LeafCNN(num_classes=NUM_CLASSES).to(device)
    state_dict = torch.load(artifact_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("LeafCNN loaded from %s (device=%s)", artifact_path, device)
    return model
