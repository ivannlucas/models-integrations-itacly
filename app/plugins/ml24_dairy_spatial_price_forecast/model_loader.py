"""Artifact loading for ml24 — three CatBoost models (h=1/3/6) plus the reference data.

Faithful to src/predict/predictor.py: model_manifest.json lists the horizons in order, each
horizon's metadata JSON carries feature_cols, selected_* blocks, the baseline weights and
ml_alpha, and the model is a native CatBoost .cbm (no scaler, no pickle). The delivered paths
inside the manifest/metadata (models\\artifacts\\...\\h1\\...) are not used: the artifact folder is
flat and files are resolved by name with ArtifactStore.path() (lazy S3 download per file).
"""
# pylint: disable=line-too-long  # repo limit is 120 (.flake8), enforced by flake8
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor

from app.infrastructure.artifact_store import ArtifactStore
from app.plugins.ml24_dairy_spatial_price_forecast import preprocessing
from app.plugins.ml24_dairy_spatial_price_forecast.constants import (
    ARTIFACT_FOLDER_NAME,
    EXOG_MONTHLY_FILENAME,
    MODEL_MANIFEST_FILENAME,
    SERIES_FILENAME,
    SPATIAL_WEIGHTS_FILENAME,
    metadata_filename,
    model_filename,
)

_store = ArtifactStore(ARTIFACT_FOLDER_NAME)


def _read_json(path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _horizon_order(manifest: dict) -> list[int]:
    return [int(item["horizon_months"]) for item in manifest["horizons"]]


def _load_models(resolve) -> dict:
    """Build {"manifest", "horizons": {h: {"model", "metadata"}}} with resolve(filename) -> path."""
    manifest = _read_json(resolve(MODEL_MANIFEST_FILENAME))
    horizons = {}
    for horizon in _horizon_order(manifest):
        metadata = _read_json(resolve(metadata_filename(horizon)))
        model = CatBoostRegressor()
        model.load_model(str(resolve(model_filename(horizon))))
        horizons[horizon] = {"model": model, "metadata": metadata}
    return {"manifest": manifest, "horizons": horizons}


def load_model_bundle() -> dict:
    """Load the delivered model_manifest.json and the three horizons (.cbm + metadata)."""
    return _load_models(_store.path)


def load_bundle_from_dir(directory: str | Path) -> dict:
    """Load a bundle written by training.save_training_artifacts (a user model from MLflow)."""
    base = Path(directory)
    return _load_models(lambda name: base / name)


def load_reference_data() -> dict:
    """Load the base series (model products only), the spatial weights and the monthly exogenous.

    Same readers as the original: pd.read_csv(..., low_memory=False) for the series,
    index_col=0 for the border matrix (load_spatial_weights), and period_dt parsed to datetime
    for the exogenous block (load_monthly_exogenous).
    """
    series = preprocessing.read_series_csv(str(_store.path(SERIES_FILENAME)))
    weights = pd.read_csv(_store.path(SPATIAL_WEIGHTS_FILENAME), index_col=0)
    exog = pd.read_csv(_store.path(EXOG_MONTHLY_FILENAME), low_memory=False)
    exog["period_dt"] = pd.to_datetime(exog["period_dt"])
    return {"series": series, "spatial_weights": weights, "exog": exog}
