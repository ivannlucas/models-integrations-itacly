"""Artifact loading for ml14 — GRU wine phytosanitary price forecast.

Faithful port of src/predict/predictor.py::_build_scalers_and_features() from the delivered
code: the input/target scalers are NOT loaded from a serialized artifact (this delivery ships
none) — they are refit at load time from the bundled reference dataset, exactly like the
original CLI does on every prediction. See inbox/a14/manifest.yaml known_issues for why this
plugin bundles the dataset instead of a frozen scaler artifact.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from app.infrastructure.artifact_store import ArtifactStore
from app.plugins.ml14_wine_phyto_price_forecast.constants import (
    ARTIFACT_FOLDER_NAME,
    DROPOUT,
    HIDDEN_SIZE,
    HORIZON_WEEKS,
    MODEL_FILENAME,
    NUM_LAYERS,
    REFERENCE_DATASET_FILENAME,
    REFERENCE_DATE_COL,
    TARGET_SERIES_COL,
    DRIFT_COL,
    TEST_RATIO,
    USER_SCALERS_FILENAME,
)
from app.plugins.ml14_wine_phyto_price_forecast.rnn_models import GRUModel

_store = ArtifactStore(ARTIFACT_FOLDER_NAME)

_TARGET_COL = "target_h"
_CURRENT_PRICE_COL = "current_price"
_DRIFT_BASELINE_COL = "drift_baseline_h"
_DELTA_TARGET_COL = "target_delta_h"
_RESIDUAL_TARGET_COL = "target_residual_h"
_EXCLUDE_COLS = {
    REFERENCE_DATE_COL, _TARGET_COL, _DELTA_TARGET_COL,
    _RESIDUAL_TARGET_COL, _CURRENT_PRICE_COL, _DRIFT_BASELINE_COL,
}


def _prepare_horizon_dataset(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    result = df.copy()
    result[_TARGET_COL] = result[TARGET_SERIES_COL].shift(-horizon)
    result[_CURRENT_PRICE_COL] = result[TARGET_SERIES_COL]
    result[_DRIFT_BASELINE_COL] = result[_CURRENT_PRICE_COL] + result[DRIFT_COL] * (horizon / 4.0)
    result[_DELTA_TARGET_COL] = result[_TARGET_COL] - result[_CURRENT_PRICE_COL]
    result[_RESIDUAL_TARGET_COL] = result[_TARGET_COL] - result[_DRIFT_BASELINE_COL]
    return result.dropna(subset=[_TARGET_COL]).copy()


def _split_temporal(df: pd.DataFrame, test_ratio: float) -> pd.DataFrame:
    fechas = sorted(df[REFERENCE_DATE_COL].unique())
    n_test = max(1, int(len(fechas) * test_ratio))
    fecha_corte = fechas[-n_test]
    return df[df[REFERENCE_DATE_COL] < fecha_corte].copy()


def _build_scalers_and_features() -> tuple[list[str], StandardScaler, StandardScaler]:
    df = pd.read_csv(_store.path(REFERENCE_DATASET_FILENAME), parse_dates=[REFERENCE_DATE_COL])
    df = df.sort_values(REFERENCE_DATE_COL).reset_index(drop=True)

    df_h = _prepare_horizon_dataset(df, HORIZON_WEEKS)
    feature_cols = [c for c in df_h.columns if c not in _EXCLUDE_COLS]
    train_df = _split_temporal(df_h, TEST_RATIO)

    input_scaler = StandardScaler()
    input_scaler.fit(train_df[feature_cols].values)

    target_scaler = StandardScaler()
    target_scaler.fit(train_df[[_RESIDUAL_TARGET_COL]].values)

    return feature_cols, input_scaler, target_scaler


def load_artifact_bundle() -> dict:
    """Load the GRU model and refit the input/target scalers from the bundled reference dataset.

    _store.path(filename) only reaches out to S3 for a given file if it isn't already present
    locally (and only if STORAGE_BUCKET is set) — matches the pattern used by other forecast
    plugins in this repo (e.g. ml16, ml23).
    """
    feature_columns, input_scaler, target_scaler = _build_scalers_and_features()

    model = GRUModel(
        input_size=len(feature_columns),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )
    state = torch.load(str(_store.path(MODEL_FILENAME)), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    return {
        "model": model,
        "feature_columns": feature_columns,
        "input_scaler_mean": np.asarray(input_scaler.mean_, dtype=np.float32),
        "input_scaler_scale": np.asarray(input_scaler.scale_, dtype=np.float32),
        "target_scaler_mean": float(target_scaler.mean_[0]),
        "target_scaler_scale": float(target_scaler.scale_[0]),
    }


def save_user_bundle(directory: Path, model, feature_columns: list[str], input_scaler, target_scaler) -> None:
    """Write a retrained model in the format load_user_bundle reads (gru_model.pt + scalers.json).

    Unlike the base model, a retrained one cannot refit its scalers from the bundled reference
    dataset (it was trained on the user's data), so they travel with it.
    """
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), directory / MODEL_FILENAME)
    with open(directory / USER_SCALERS_FILENAME, "w", encoding="utf-8") as fh:
        json.dump({
            "feature_columns": list(feature_columns),
            "input_mean": input_scaler.mean_.tolist(), "input_scale": input_scaler.scale_.tolist(),
            "target_mean": float(target_scaler.mean_[0]), "target_scale": float(target_scaler.scale_[0]),
        }, fh, indent=2)


def load_user_bundle(directory: str | Path) -> dict:
    """Load a bundle written by save_user_bundle — same keys as load_artifact_bundle."""
    base = Path(directory)
    with open(base / USER_SCALERS_FILENAME, encoding="utf-8") as fh:
        meta = json.load(fh)
    model = GRUModel(input_size=len(meta["feature_columns"]), hidden_size=HIDDEN_SIZE,
                     num_layers=NUM_LAYERS, dropout=DROPOUT)
    model.load_state_dict(torch.load(base / MODEL_FILENAME, map_location="cpu", weights_only=True))
    model.eval()
    return {
        "model": model,
        "feature_columns": list(meta["feature_columns"]),
        "input_scaler_mean": np.asarray(meta["input_mean"], dtype=np.float32),
        "input_scaler_scale": np.asarray(meta["input_scale"], dtype=np.float32),
        "target_scaler_mean": float(meta["target_mean"]),
        "target_scaler_scale": float(meta["target_scale"]),
    }
