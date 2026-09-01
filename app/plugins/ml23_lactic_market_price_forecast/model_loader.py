"""Artifact loading for ml23 — GRU dairy price forecast."""
from __future__ import annotations

import json

import numpy as np
import torch

from app.infrastructure.artifact_store import ArtifactStore
from app.plugins.ml23_lactic_market_price_forecast.constants import (
    ARTIFACT_FOLDER_NAME,
    MANIFEST_FILENAME,
    MODEL_FILENAME,
    SCALER_FILENAME,
)
from app.plugins.ml23_lactic_market_price_forecast.rnn_models import GRUModel

_store = ArtifactStore(ARTIFACT_FOLDER_NAME)


def load_model_bundle() -> tuple[GRUModel, np.ndarray, np.ndarray, dict]:
    """Return (model, mean, scale, manifest), downloading each artifact lazily if missing.

    _store.path(filename) only reaches out to S3 for a given file if it isn't already
    present locally (and only if STORAGE_BUCKET is set) -- unlike the previous
    _store.download_all_if_needed() call this replaces, which raised EnvironmentError
    immediately whenever STORAGE_BUCKET was unset, even with all 3 artifacts already
    vendored locally under artifacts/ml23_lactic_market_price_forecast/. That broke
    load() in every local/dev/CI environment without S3 configured (see
    inbox/a23/manifest.yaml known_issues) -- same pattern already used correctly by
    most other plugins in this repo (e.g. ml16_meat_raw_material_price_alert).
    """
    with open(_store.path(MANIFEST_FILENAME), encoding="utf-8") as fh:
        manifest = json.load(fh)

    bundle = np.load(str(_store.path(SCALER_FILENAME)))
    scaler_mean: np.ndarray = bundle["mean"]
    scaler_scale: np.ndarray = bundle["scale"]

    input_size = len(manifest["feature_cols"])
    hidden_size = int(manifest["hidden_size"])
    model = GRUModel(input_size=input_size, hidden_size=hidden_size)
    state = torch.load(str(_store.path(MODEL_FILENAME)), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    return model, scaler_mean, scaler_scale, manifest
