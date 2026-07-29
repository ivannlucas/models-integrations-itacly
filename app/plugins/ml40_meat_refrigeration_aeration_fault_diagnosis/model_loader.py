"""Loads ml40 artifacts (per-system RandomForest + scaler + thresholds) via ArtifactStore."""
from __future__ import annotations

import logging

import joblib
import yaml

from app.infrastructure.artifact_store import ArtifactStore
from app.plugins.ml40_meat_refrigeration_aeration_fault_diagnosis.constants import (
    ARTIFACT_FOLDER_NAME,
    MODEL_FILENAMES,
    SCALER_FILENAMES,
    STATS_FILENAMES,
    SYSTEMS,
    THRESHOLDS_FILENAMES,
)

logger = logging.getLogger(__name__)

_store = ArtifactStore(ARTIFACT_FOLDER_NAME)


def _load_yaml(path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_system_artifacts(system: str) -> dict:
    """Load model, optional scaler, neurosymbolic thresholds and drift stats for one system."""
    bundle = {
        "model": joblib.load(_store.path(MODEL_FILENAMES[system])),
        "scaler": None,
        "thresholds": _load_yaml(_store.path(THRESHOLDS_FILENAMES[system])),
        "stats": _load_yaml(_store.path(STATS_FILENAMES[system])),
    }
    if system in SCALER_FILENAMES:
        bundle["scaler"] = joblib.load(_store.path(SCALER_FILENAMES[system]))
    logger.info(
        "ml40 artifacts loaded for %s — model=%s thresholds=%s",
        system, type(bundle["model"]).__name__, sorted(bundle["thresholds"]),
    )
    return bundle


def load_artifacts() -> dict[str, dict]:
    """Load both subsystems. The refrigeration RandomForest is ~171 MB on disk."""
    return {system: load_system_artifacts(system) for system in SYSTEMS}
