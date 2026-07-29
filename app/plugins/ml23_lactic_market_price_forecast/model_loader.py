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
    """Download artifacts if needed and return (model, mean, scale, manifest)."""
    _store.download_all_if_needed()

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
