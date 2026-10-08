"""Loads ml13 production artifacts (model + StandardScaler + feature schema + model config) via
ArtifactStore — same files as the delivered models/prod/."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib

from app.infrastructure.artifact_store import ArtifactStore
from app.plugins.ml13_wine_price_fluctuation_prediction.constants import (
    ARTIFACT_FOLDER_NAME,
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_FILENAME,
    MODEL_CONFIG_FILENAME,
    MODEL_FILENAME,
    SCALER_FILENAME,
)

logger = logging.getLogger(__name__)

_store = ArtifactStore(ARTIFACT_FOLDER_NAME)


def load_bundle_from_paths(
    model_path: Path, scaler_path: Path, schema_path: Path, config_path: Path,
) -> dict:
    """Load one artifact bundle from explicit paths (shared by fixed and MLflow user bundles)."""
    with open(schema_path, encoding="utf-8") as fh:
        schema = json.load(fh)
    with open(config_path, encoding="utf-8") as fh:
        model_config = json.load(fh)
    feature_columns = schema.get("feature_columns", FEATURE_COLUMNS)
    if list(feature_columns) != FEATURE_COLUMNS:
        raise ValueError(
            f"feature_schema.json declares {feature_columns}, expected {FEATURE_COLUMNS} "
            "(the plugin's feature engineering only produces these columns)."
        )
    model_type = model_config.get("model_type", "logreg")
    if model_type not in {"logreg", "xgboost"}:
        raise ValueError(f"Unknown model_type in {config_path.name}: {model_type}")
    return {
        "model": joblib.load(model_path),
        "scaler": joblib.load(scaler_path),
        "schema": schema,
        "model_type": model_type,
    }


def load_artifacts() -> dict:
    """Load the fixed (AI-team) bundle from artifacts/<ARTIFACT_FOLDER_NAME>/ or S3."""
    bundle = load_bundle_from_paths(
        _store.path(MODEL_FILENAME),
        _store.path(SCALER_FILENAME),
        _store.path(FEATURE_SCHEMA_FILENAME),
        _store.path(MODEL_CONFIG_FILENAME),
    )
    logger.info("ml13 artifacts loaded — model_type=%s", bundle["model_type"])
    return bundle
