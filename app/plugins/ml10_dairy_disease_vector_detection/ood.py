"""«desconocido» for ml10: detections whose crop does not look like a fly, mosquito or tick.

ml10 is detect → crop → classify, and neither stage has a "none of these" class. On photos of
another domain (vine leaves, poultry droppings, cows, grain, charts…) the detector still fires
at det_conf ≥ 0.2 and the classifier labels the crop fly/mos/tick at ~0.9, the same mean
confidence as on real a10 test images, so a confidence threshold cannot separate them
(outputs/a10/README_ood.md).

The gate (app/domain/services/feature_ood.py, shared with ml8) scores every crop that passes
cls_conf_thresh with the classifier's own 960-d pooled MobileNetV3 features (the input of its
head). Its bank holds the classifier's training crops plus the crops the detector takes from real
a10 detection images that match a labelled box: with the training crops alone ~55 % of real
detector crops were flagged, because the pre-cropped ImageFolder images look different from what
the detector cuts. The threshold keeps 95 % of held-out real crops. Above it the detection's
species becomes «desconocido» and the network's label moves to ``species_modelo``; the
classifier, its confidences and every in-distribution prediction are unchanged.

``ood_crops.zip`` (next to the classifier) holds the detector crops (``id_train/``, ``id_val/``)
and the outlier-domain crops (``outliers/<dominio>/``), so /train rebuilds the gate for a
retrained classifier: training changes its BatchNorm statistics, and with them the features.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from app.domain.services.exceptions import InvalidImageError
from app.domain.services.feature_ood import (
    UNKNOWN_LABEL,
    OodReference,
    build_reference,
)
from app.plugins.ml10_dairy_disease_vector_detection.constants import SUPPORTED_EXTENSIONS
from app.plugins.ml10_dairy_disease_vector_detection.preprocessing import (
    CLASSIFIER_TRANSFORM,
    image_path_to_pil,
)

OOD_CROPS_FILENAME = "ood_crops.zip"
# Folders of ood_crops.zip.
ID_TRAIN_DIR = "id_train"
ID_VAL_DIR = "id_val"
OUTLIERS_DIR = "outliers"
_FEATURE_DIM = 960
_FEATURE_BATCH = 32


def pooled_features(classifier: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """The 960-d pooled MobileNetV3 features (torchvision forward up to the classifier head)."""
    return torch.flatten(classifier.avgpool(classifier.features(x)), 1)


def classify_crop_with_features(
    classifier: torch.nn.Module, tensor: torch.Tensor, class_names: list[str], device
) -> tuple[str, float, np.ndarray]:
    """``(species, confidence, features)`` of one preprocessed crop.

    Same computation as ``classifier(tensor)`` (features → avgpool → flatten → classifier),
    also returning the pooled features the gate scores.
    """
    with torch.no_grad():
        feats = pooled_features(classifier, tensor.to(device))
        probs = torch.softmax(classifier.classifier(feats), dim=1)[0]
    idx = int(torch.argmax(probs).item())
    return class_names[idx], float(probs[idx].item()), feats.cpu().numpy()


def apply_gate(
    detection: dict, reference: OodReference | None, features: np.ndarray | None
) -> dict:
    """Add the «desconocido» fields to a detection dict (in place) and return it.

    ``species`` becomes «desconocido» above the threshold and the network's label is kept in
    ``species_modelo``; ``cls_conf`` stays the network's confidence. Without a reference (base
    artifact without ood_reference.npz, or a run retrained before the gate) the detection is
    returned as before, with ``es_desconocido`` False and a null score.
    """
    detection["species_modelo"] = detection["species"]
    if reference is None or features is None:
        detection.update(es_desconocido=False, puntuacion_ood=None)
        return detection
    score = float(reference.score(features)[0])
    unknown = score > reference.threshold
    detection.update(es_desconocido=bool(unknown), puntuacion_ood=round(score, 6))
    if unknown:
        detection["species"] = UNKNOWN_LABEL
    return detection


def image_files(folder: Path) -> list[Path]:
    """Image files under *folder* (recursive, sorted); empty if it does not exist."""
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.rglob("*")
                  if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)


def crop_features(classifier: torch.nn.Module, paths: Iterable, device) -> np.ndarray:
    """Pooled features of crop image files with the inference preprocessing (unreadable: skipped).

    The crops go through ``CLASSIFIER_TRANSFORM``, the transform every detector crop gets at
    inference, so the bank and the threshold describe what the gate will actually score.
    """
    classifier.eval()
    feats: list[np.ndarray] = []
    batch: list[torch.Tensor] = []

    def _flush() -> None:
        if batch:
            with torch.no_grad():
                x = torch.stack(batch).to(device)
                feats.append(pooled_features(classifier, x).cpu().numpy())
            batch.clear()

    for path in paths:
        try:
            batch.append(CLASSIFIER_TRANSFORM(image_path_to_pil(str(path))))
        except (InvalidImageError, OSError, ValueError):
            continue
        if len(batch) == _FEATURE_BATCH:
            _flush()
    _flush()
    return np.concatenate(feats) if feats else np.zeros((0, _FEATURE_DIM), np.float32)


def build_crops_reference(
    classifier: torch.nn.Module,
    device,
    train_paths: list,
    val_paths: list,
    crops_dir: Path,
) -> OodReference:
    """The gate of *classifier*: bank, threshold and regression from its own features.

    * bank: the classifier's training crops + ``id_train/`` (real detector crops);
    * threshold: 95 % of the validation crops + ``id_val/`` kept (none of them in the bank);
    * regression: bank vs ``outliers/`` (other domains).

    The base reference (outputs/a10/generar_referencia_ood.py) and /train use this same function.
    """
    bank = np.r_[crop_features(classifier, train_paths, device),
                 crop_features(classifier, image_files(crops_dir / ID_TRAIN_DIR), device)]
    held_out = np.r_[crop_features(classifier, val_paths, device),
                     crop_features(classifier, image_files(crops_dir / ID_VAL_DIR), device)]
    outliers = crop_features(classifier, image_files(crops_dir / OUTLIERS_DIR), device)
    return build_reference(bank, held_out, outliers if len(outliers) else None)
