"""Loads the cereal residue-optimizer surrogate (sklearn Pipeline) via ArtifactStore."""
import logging

import joblib

from app.infrastructure.artifact_store import ArtifactStore
from app.plugins.ml31_cereals_residue_optimizer.constants import (
    ARTIFACT_FOLDER_NAME,
    MODEL_FILENAME,
)

logger = logging.getLogger(__name__)

_store = ArtifactStore(ARTIFACT_FOLDER_NAME)


def load_pipeline():
    """Load and return the fitted sklearn Pipeline (downloads from S3 if configured)."""
    pipe = joblib.load(_store.path(MODEL_FILENAME))
    logger.info("Ml31CerealsResidueOptimizer pipeline loaded")
    return pipe
