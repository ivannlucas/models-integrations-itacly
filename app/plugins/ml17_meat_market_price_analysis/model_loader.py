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
    """Return the fitted Ridge estimator, downloading it lazily if needed.

    _store.path(filename) only reaches out to S3 if the file isn't already present
    locally (and only if STORAGE_BUCKET is set) — unlike the unconditional
    _store.download_all_if_needed() this used to call, which raised EnvironmentError
    immediately whenever STORAGE_BUCKET was unset, even with the artifact already
    vendored locally (same bug found and fixed for ml23; see
    inbox/a23/manifest.yaml known_issues).
    """
    return joblib.load(str(_store.path(MODEL_FILENAME)))
