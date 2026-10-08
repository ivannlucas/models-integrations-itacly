"""Artifact loading for ml18 — GRU meat spatial price forecast.

Faithful port of src/predict/predictor.py::load_model_artifacts() from the delivered code: the
Keras model is NOT saved in native .keras/.h5 format — it is a joblib-wrapped dict
({'model_json': model.to_json(), 'weights': model.get_weights()}), reconstructed via
tf.keras.models.model_from_json() + set_weights(). The input/target scalers ARE frozen,
versioned JSON artifacts (mean/scale/var) — no live refit needed, unlike ml14.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from app.infrastructure.artifact_store import ArtifactStore
from app.plugins.ml18_meat_spatial_price_forecast.constants import (
    ARTIFACT_FOLDER_NAME,
    MODEL_FILENAME,
    X_SCALER_FILENAME,
    Y_SCALER_FILENAME,
)

_store = ArtifactStore(ARTIFACT_FOLDER_NAME)


def _load_scaler_from_json(path) -> StandardScaler:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    scaler = StandardScaler()
    scaler.mean_ = np.array(data["mean"])
    scaler.scale_ = np.array(data["scale"])
    scaler.var_ = np.array(data["var"])
    return scaler


def load_artifact_bundle() -> dict:
    """Load the GRU model bundle and the frozen input/target scalers.

    _store.path(filename) only reaches out to S3 for a given file if it isn't already present
    locally (and only if STORAGE_BUCKET is set) — matches the pattern used by other forecast
    plugins in this repo (e.g. ml14, ml16, ml23).
    """
    # TensorFlow is imported lazily so the plugin module stays light for unit tests that never
    # touch the real model (FakePlugin-based wiring tests).
    # pylint: disable=import-outside-toplevel
    import tensorflow as tf

    model_bundle = joblib.load(_store.path(MODEL_FILENAME))
    model = tf.keras.models.model_from_json(model_bundle["model_json"])
    model.set_weights(model_bundle["weights"])

    x_scaler = _load_scaler_from_json(_store.path(X_SCALER_FILENAME))
    y_scaler = _load_scaler_from_json(_store.path(Y_SCALER_FILENAME))

    return {"model": model, "x_scaler": x_scaler, "y_scaler": y_scaler}
