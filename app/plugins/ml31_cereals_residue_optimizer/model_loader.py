"""Loads the ml31 LP-optimizer reference artifacts via ArtifactStore.

There are no serialized model weights: the artifacts are the economic reference
data (crop_economics.json / harvest_index.json) and the historical dataset CSV.
"""
import json
import logging

import pandas as pd

from app.infrastructure.artifact_store import ArtifactStore
from app.plugins.ml31_cereals_residue_optimizer.constants import (
    ARTIFACT_FOLDER_NAME,
    CROP_ECONOMICS_FILENAME,
    DATASET_FILENAME,
    HARVEST_INDEX_FILENAME,
)

logger = logging.getLogger(__name__)

_store = ArtifactStore(ARTIFACT_FOLDER_NAME)


def load_artifacts():
    """Load the economic/agronomic reference data and the historical dataset.

    Returns (economics_dict, harvest_indices_dict, dataframe). Downloads from S3
    if configured and files are missing locally.
    """
    with open(_store.path(CROP_ECONOMICS_FILENAME), "r", encoding="utf-8") as f:
        economics = json.load(f)
    with open(_store.path(HARVEST_INDEX_FILENAME), "r", encoding="utf-8") as f:
        harvest_indices = json.load(f)
    df = pd.read_csv(_store.path(DATASET_FILENAME))

    logger.info(
        "ml31 artifacts loaded — %d crops, dataset %d rows (%d-%d)",
        len(economics.get("crops", {})),
        len(df),
        int(df["Año"].min()),
        int(df["Año"].max()),
    )
    return economics, harvest_indices, df
