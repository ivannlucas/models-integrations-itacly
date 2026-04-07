import json
import logging
from typing import Any

import joblib

from app.infrastructure.artifact_store import ArtifactStore

logger = logging.getLogger(__name__)

_store = ArtifactStore("wine_sulphite")


def load_artifacts() -> tuple[Any, Any, dict]:
    quality_path = _store.path("quality_rf.pkl")
    bound_path = _store.path("bound_rf.pkl")
    metadata_path = _store.path("metadata.json")

    logger.info("Loading quality model from %s", quality_path)
    model_qual = joblib.load(quality_path)

    logger.info("Loading bound SO2 model from %s", bound_path)
    model_bound = joblib.load(bound_path)

    logger.info("Loading metadata from %s", metadata_path)
    with open(metadata_path) as f:
        metadata = json.load(f)

    logger.info(
        "All artifacts loaded — quality MAE=%.3f, bound MAE=%.3f",
        metadata.get("metrics", {}).get("quality_cv", {}).get("mae_mean", float("nan")),
        metadata.get("metrics", {}).get("bound_cv", {}).get("mae_mean", float("nan")),
    )
    return model_qual, model_bound, metadata
