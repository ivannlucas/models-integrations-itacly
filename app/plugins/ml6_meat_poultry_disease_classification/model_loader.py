"""Builds the six base models and loads the AI team's checkpoints via ArtifactStore.

The wrappers mirror src/training/<model>.py attribute by attribute, because the delivered
state_dicts are keyed by them (``densenet121.features...``, ``efficientnetv2s.features...``).
The originals build the backbone with ``weights='IMAGENET1K_V1'`` (a download even for
inference); here it is built with ``weights=None`` and the checkpoint is loaded with
``strict=True``, so every tensor comes from the delivered file.
"""
import logging
from typing import Callable

import torch
from torch import nn
from torchvision import models

from app.infrastructure.artifact_store import ArtifactStore
from app.plugins.ml6_meat_poultry_disease_classification.constants import (
    ARTIFACT_FOLDER_NAME,
    CLASS_NAMES,
    MODEL_FILENAMES,
    TASK_MODELS,
)

logger = logging.getLogger(__name__)

_store = ArtifactStore(ARTIFACT_FOLDER_NAME)


def _freeze(module: nn.Module) -> None:
    """Freeze every parameter, as the originals do before replacing the head."""
    for param in module.parameters():
        param.requires_grad = False


class DenseNet121(nn.Module):
    """src/training/densenet121.py"""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.densenet121 = models.densenet121(weights=None)
        _freeze(self.densenet121)
        in_features = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """Return the class logits."""
        return self.densenet121(x)


class EfficientNetV2S(nn.Module):
    """src/training/efficientnet_v2s.py"""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.efficientnetv2s = models.efficientnet_v2_s(weights=None)
        _freeze(self.efficientnetv2s)
        in_features = self.efficientnetv2s.classifier[1].in_features
        self.efficientnetv2s.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """Return the class logits."""
        return self.efficientnetv2s(x)


class ResNet50(nn.Module):
    """src/training/resnet50.py"""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.resnet50 = models.resnet50(weights=None)
        _freeze(self.resnet50)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, num_classes)

    def forward(self, x):
        """Return the class logits."""
        return self.resnet50(x)


class VGG19BN(nn.Module):
    """src/training/vgg19_bn.py"""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.vgg19_bn = models.vgg19_bn(weights=None)
        _freeze(self.vgg19_bn)
        in_features = self.vgg19_bn.classifier[6].in_features
        self.vgg19_bn.classifier[6] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """Return the class logits."""
        return self.vgg19_bn(x)


class EfficientNetB7(nn.Module):
    """src/training/efficientnet_b7.py"""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.efficientnet_b7 = models.efficientnet_b7(weights=None)
        _freeze(self.efficientnet_b7)
        in_features = self.efficientnet_b7.classifier[1].in_features
        self.efficientnet_b7.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """Return the class logits."""
        return self.efficientnet_b7(x)


class MobileNetV3L(nn.Module):
    """src/training/mobilenet_v3l.py"""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.mobilenet_v3 = models.mobilenet_v3_large(weights=None)
        _freeze(self.mobilenet_v3)
        in_features = self.mobilenet_v3.classifier[3].in_features
        self.mobilenet_v3.classifier[3] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """Return the class logits."""
        return self.mobilenet_v3(x)


BUILDERS: dict[str, Callable[[int], nn.Module]] = {
    "densenet121": DenseNet121,
    "efficientnet_v2s": EfficientNetV2S,
    "resnet50": ResNet50,
    "vgg19_bn": VGG19BN,
    "efficientnet_b7": EfficientNetB7,
    "mobilenet_v3l": MobileNetV3L,
}


def safe_device() -> torch.device:
    """Return CUDA only if a conv actually runs on it (module AND tensor on the GPU)."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        probe = torch.nn.Conv2d(1, 1, 1).cuda()
        probe(torch.zeros(1, 1, 4, 4).cuda())
        return torch.device("cuda")
    except Exception:  # pylint: disable=broad-except
        logger.warning("CUDA detectada pero no funcional — usando CPU.")
        return torch.device("cpu")


def build_from_checkpoint(name: str, task: str, checkpoint_path, device: torch.device) -> nn.Module:
    """Build base model ``name`` of ``task`` and load ``checkpoint_path`` (strict) in eval mode."""
    model = BUILDERS[name](len(CLASS_NAMES[task]))
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def load_ensembles() -> tuple[dict[str, dict[str, nn.Module]], torch.device]:
    """Load both ensembles from the base artifact folder: ``{task: {name: model}}``."""
    device = safe_device()
    ensembles = {
        task: {
            name: build_from_checkpoint(name, task, _store.path(MODEL_FILENAMES[name]), device)
            for name in names
        }
        for task, names in TASK_MODELS.items()
    }
    logger.info("ml6 ensembles loaded (device=%s)", device)
    return ensembles, device
