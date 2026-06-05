"""Model artifact loader for the wine sulphite plugin."""
import json
import logging
from typing import Any

import joblib

from app.infrastructure.artifact_store import ArtifactStore
from app.plugins.ml25_wine_sulphites.constants import (
    ARTIFACT_FOLDER_NAME,
    QUALITY_RF_MODEL_FILENAME,
    BOUND_RF_MODEL_FILENAME,
    METADATA_FILENAME,
)

logger = logging.getLogger(__name__)

_store = ArtifactStore(ARTIFACT_FOLDER_NAME)


def get_artifacts_dir():
    """Return the local directory path where wine-sulphite artifacts are stored."""
    return _store.get_local_dir()


def upload_artifact(filename: str) -> None:
    """Upload *filename* from the local artifact directory to S3 via the artifact store."""
    _store.upload(filename)


def load_artifacts() -> tuple[Any, Any, dict]:
    """Load the quality model, bound SO2 model, and metadata from disk (or S3).

    quality_path = _store.path(QUALITY_RF_MODEL_FILENAME)
    bound_path = _store.path(BOUND_RF_MODEL_FILENAME)
    metadata_path = _store.path(METADATA_FILENAME)

    logger.info("Loading quality model from %s", quality_path)
    model_qual = joblib.load(quality_path)

    logger.info("Loading bound SO2 model from %s", bound_path)
    model_bound = joblib.load(bound_path)

    logger.info("Loading metadata from %s", metadata_path)
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    logger.info(
        "All artifacts loaded — quality MAE=%.3f, bound MAE=%.3f",
        metadata.get("metrics", {}).get("quality_cv", {}).get("mae_mean", float("nan")),
        metadata.get("metrics", {}).get("bound_cv", {}).get("mae_mean", float("nan")),
    )
    return model_qual, model_bound, metadata
