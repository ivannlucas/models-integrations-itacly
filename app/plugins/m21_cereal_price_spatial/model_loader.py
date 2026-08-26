"""Artifact loading for m21 — ESP-CEREAL spatial cereal price prediction."""
from __future__ import annotations

import json
import logging
from typing import Any

import joblib

from app.infrastructure.artifact_store import ArtifactStore
from app.plugins.m21_cereal_price_spatial.constants import (
    ARTIFACT_FOLDER_NAME,
    METADATA_FILENAME,
    MODEL_H1_CLF,
    MODEL_H1_REG,
    MODEL_H2_CLF,
    MODEL_H2_REG,
    MODEL_H3_CLF,
    MODEL_H3_REG,
)

logger = logging.getLogger(__name__)

_store = ArtifactStore(ARTIFACT_FOLDER_NAME)


def load_model_bundle() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Download artifacts if needed and return (models_by_horizon, metadata).

    models_by_horizon = {
        "H1": {"reg": model, "clf": model},
        "H2": {"reg": model, "clf": model},
        "H3": {"reg": model, "clf": model},
    }
    """
    _store.download_all_if_needed()

    with open(_store.path(METADATA_FILENAME), encoding="utf-8") as fh:
        metadata = json.load(fh)

    model_files = {
        "H1": {"reg": MODEL_H1_REG, "clf": MODEL_H1_CLF},
        "H2": {"reg": MODEL_H2_REG, "clf": MODEL_H2_CLF},
        "H3": {"reg": MODEL_H3_REG, "clf": MODEL_H3_CLF},
    }

    models: dict[str, dict[str, Any]] = {}
    for h_key, files in model_files.items():
        reg_path = _store.path(files["reg"])
        clf_path = _store.path(files["clf"])

        if not reg_path.exists() or not clf_path.exists():
            raise FileNotFoundError(
                f"Modelos no encontrados para {h_key}: {reg_path}, {clf_path}"
            )

        models[h_key] = {
            "reg": joblib.load(reg_path),
            "clf": joblib.load(clf_path),
        }

    logger.info(
        "m21 loaded — %s (reg=%s, clf=%s), %s (reg=%s, clf=%s), %s (reg=%s, clf=%s)",
        "H1", type(models["H1"]["reg"]).__name__, type(models["H1"]["clf"]).__name__,
        "H2", type(models["H2"]["reg"]).__name__, type(models["H2"]["clf"]).__name__,
        "H3", type(models["H3"]["reg"]).__name__, type(models["H3"]["clf"]).__name__,
    )
    return models, metadata
