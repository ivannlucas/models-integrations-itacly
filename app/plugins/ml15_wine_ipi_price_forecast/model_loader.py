"""Loads the ml15 artifact (joblib payload: Pipeline(StandardScaler+Ridge) + feature contract)
via ArtifactStore.
"""
from __future__ import annotations

import logging

import joblib

from app.infrastructure.artifact_store import ArtifactStore
from app.plugins.ml15_wine_ipi_price_forecast.constants import ARTIFACT_FOLDER_NAME, MODEL_FILENAME

logger = logging.getLogger(__name__)

_store = ArtifactStore(ARTIFACT_FOLDER_NAME)

_REQUIRED_PAYLOAD_KEYS = ("model", "feature_columns", "target_column", "anchor_column")


def load_artifact() -> dict:
    """Load the horizon_6_ridge.pkl payload.

    The payload is the dict persisted by the AI team's trainer.py: {model, feature_columns,
    target_column, anchor_column, horizon, model_name, model_kind, model_params, ...}. The
    plugin trusts this payload as the source of truth for feature_columns/target_column/
    anchor_column at runtime (not the constants.py copies, which exist only for DTO docs and
    for the fixed schema train() rebuilds artifacts with).
    """
    payload = joblib.load(_store.path(MODEL_FILENAME))
    missing = [key for key in _REQUIRED_PAYLOAD_KEYS if key not in payload]
    if missing:
        raise ValueError(f"Artefacto {MODEL_FILENAME} inválido, faltan claves: {missing}")
    logger.info(
        "ml15 artifact loaded — horizon=%s model=%s n_features=%d",
        payload.get("horizon"), payload.get("model_name"), len(payload["feature_columns"]),
    )
    return payload
