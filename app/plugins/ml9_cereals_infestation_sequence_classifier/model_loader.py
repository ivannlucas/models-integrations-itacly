"""Loads the ml9 sequence-classifier artifacts (GRU checkpoint + scaler + bundle) via ArtifactStore."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.infrastructure.artifact_store import ArtifactStore
from app.plugins.ml9_cereals_infestation_sequence_classifier._vendor.sequential import load_checkpoint, load_pickle
from app.plugins.ml9_cereals_infestation_sequence_classifier.constants import (
    ARTIFACT_FOLDER_NAME,
    BUNDLE_FILENAME,
    MODEL_FILENAME,
    SCALER_FILENAME,
)

logger = logging.getLogger(__name__)

_store = ArtifactStore(ARTIFACT_FOLDER_NAME)

# Runtime skeleton config. Every section that affects model semantics (data, sequence,
# synthetic_dataset, training) is overwritten from the bundle's config_snapshot by
# merge_runtime_config_with_bundle() — exactly like the delivered predictor does. `paths` is kept
# only because the vendored helpers read cfg["paths"] nowhere in the inference path; inference is
# fully in-memory, so no directory is ever created.
RUNTIME_CFG: dict[str, Any] = {
    "paths": {},
    "training": {"device": "cpu", "fallback_to_cpu": True, "quick": False},
}


def _load_json(path: str | Path) -> dict:
    """Read a JSON file as UTF-8."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_bundle() -> dict:
    """Load model_bundle_metadata.json (feature order, window params, config snapshot, winner)."""
    return _load_json(_store.path(BUNDLE_FILENAME))


def load_artifacts() -> tuple[dict, Any, dict]:
    """Load the served checkpoint, the fitted StandardScaler and the training bundle.

    Returns (checkpoint, scaler, bundle). `checkpoint["model"]` is the ready-to-use
    SequenceClassifier in eval mode (built by the vendored load_checkpoint()).
    """
    bundle = load_bundle()
    checkpoint = load_checkpoint(_store.path(MODEL_FILENAME), map_location="cpu")
    scaler = load_pickle(_store.path(SCALER_FILENAME))

    logger.info(
        "ml9 artifacts loaded — winner=%s model_type=%s n_features=%d window_size=%s stride=%s",
        bundle.get("best_model_name"),
        checkpoint.get("model_type"),
        len(bundle.get("feature_columns") or []),
        bundle.get("window_size"),
        bundle.get("stride"),
    )
    return checkpoint, scaler, bundle
