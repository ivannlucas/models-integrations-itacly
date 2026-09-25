"""Artifact loading for ml14 — LSTM wine phytosanitary price forecast."""
from __future__ import annotations

import json

import torch

from app.infrastructure.artifact_store import ArtifactStore
from app.plugins.ml14_wine_phyto_price_forecast.constants import (
    ARTIFACT_FOLDER_NAME,
    DROPOUT,
    FEATURES_FILENAME,
    HIDDEN_SIZE,
    MODEL_FILENAME,
    NUM_LAYERS,
    PREPROCESSING_FILENAME,
)
from app.plugins.ml14_wine_phyto_price_forecast.rnn_models import LSTMModel

_store = ArtifactStore(ARTIFACT_FOLDER_NAME)


def load_artifact_bundle() -> dict:
    """Load the LSTM model, its frozen scalers and the binding feature-order contract.

    _store.path(filename) only reaches out to S3 for a given file if it isn't already
    present locally (and only if STORAGE_BUCKET is set) — matches the pattern used by other
    forecast plugins in this repo (e.g. ml16, ml23); avoids requiring STORAGE_BUCKET in
    local/dev/CI environments where the 3 artifacts are already vendored under
    artifacts/ml14_wine_phyto_price_forecast/.
    """
    with open(_store.path(FEATURES_FILENAME), encoding="utf-8") as fh:
        features_contract = json.load(fh)
    feature_columns: list[str] = list(features_contract["features"])

    with open(_store.path(PREPROCESSING_FILENAME), encoding="utf-8") as fh:
        preprocessing = json.load(fh)

    input_scaler = preprocessing["input_scaler"]
    target_scaler = preprocessing["target_scaler"]

    model = LSTMModel(
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
        "input_scaler_mean": input_scaler["mean"],
        "input_scaler_scale": input_scaler["scale"],
        "target_scaler_mean": target_scaler["mean"][0],
        "target_scaler_scale": target_scaler["scale"][0],
        "preprocessing": preprocessing,
    }
