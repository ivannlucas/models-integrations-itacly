"""Artifact loading for ml17 — Ridge pork price model."""
from __future__ import annotations

import joblib

from app.infrastructure.artifact_store import ArtifactStore
from app.plugins.ml17_meat_market_price_analysis.constants import (
    ARTIFACT_FOLDER_NAME,
    MODEL_FILENAME,
)

_store = ArtifactStore(ARTIFACT_FOLDER_NAME)


def load_model() -> object:
    """Download artifact if needed and return the fitted Ridge estimator."""
    _store.download_all_if_needed()
    return joblib.load(str(_store.path(MODEL_FILENAME)))
