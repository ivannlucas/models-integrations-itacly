"""Carga de modelos para el plugin Modelo10Lacteo."""
from __future__ import annotations

import json
import logging

import torch
from torch import nn
from torchvision import models

from app.infrastructure.artifact_store import ArtifactStore
from app.plugins.modelo10_lacteo.constants import (
    ARTIFACT_FOLDER_NAME,
    CLASSIFIER_FILENAME,
    CLASS_NAMES_FILENAME,
    DETECTOR_FILENAME
)

logger = logging.getLogger(__name__)

SPECIES = ["fly", "mos", "tick"]

_store = ArtifactStore(ARTIFACT_FOLDER_NAME)


def _build_mobilenetv3_classifier(num_classes: int) -> nn.Module:
    """MobileNetV3-Large con cabeza de clasificación reemplazada."""
    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def load_detector_and_classifier(device: torch.device):
    """
    Carga el detector YOLO y el clasificador MobileNetV3 desde artifacts/.
    Devuelve (detector, classifier, class_names).
    """
    _store.download_all_if_needed()

    detector_path = _store.path(DETECTOR_FILENAME)
    classifier_path = _store.path(CLASSIFIER_FILENAME)
    class_names_path = _store.path(CLASS_NAMES_FILENAME)

    for p in (detector_path, classifier_path, class_names_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Artefacto no encontrado: {p}. "
                "Copia los ficheros en artifacts/ antes de arrancar el servicio."
            )

    # Load class names
    with open(class_names_path, "r") as fh:
        class_names: list[str] = json.load(fh)

    # Load YOLO detector (lazy import to avoid heavy ultralytics import at module level)
    from ultralytics import YOLO  # noqa: PLC0415
    detector = YOLO(str(detector_path))

    # Load MobileNetV3 classifier
    classifier = _build_mobilenetv3_classifier(len(class_names))
    state_dict = torch.load(str(classifier_path), map_location=device)
    classifier.load_state_dict(state_dict)
    classifier.eval()
    classifier.to(device)

    logger.info(
        "Modelo10Lacteo cargado — detector: %s, clasificador: %s, clases: %s",
        detector_path.name, classifier_path.name, class_names,
    )
    return detector, classifier, class_names
