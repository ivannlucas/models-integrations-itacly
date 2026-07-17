"""MLflow helper for ml31 cereal residue optimizer.

Mandatory by repo convention even though the v2.0 model is a deterministic LP
optimizer with NO user retraining (training.supported=false) and NO serialized
weights. There is no user-trained artifact to fetch, so this always returns
None; predict/stats fall back to the fixed reference data loaded by
model_loader.py. Kept for interface uniformity across the 48 plugins.
"""
from __future__ import annotations

import json
import logging
import os

import pandas as pd

from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.plugins.ml31_cereals_residue_optimizer.constants import (
    CROP_ECONOMICS_FILENAME,
    DATASET_FILENAME,
    HARVEST_INDEX_FILENAME,
)

logger = logging.getLogger(__name__)


def download_user_model_from_mlflow(run_id: str):
    """Attempt to download user-supplied reference data from MLflow.

    Returns (economics, harvest_indices, df, temp_dir) or None. The v2.0 model is
    not retrained by users, so in practice this returns None and the caller uses
    the fixed reference artifacts. Caller MUST shutil.rmtree(temp_dir) after use
    (try/finally) if a non-None result is returned.
    """
    import tempfile
    tmp = tempfile.mkdtemp(prefix="mlflow_ml31_")
    local_path = BaseMLflowTracker(run_id).download_artifacts(tmp, artifact_path="model")
    if not local_path:
        return None

    try:
        with open(os.path.join(local_path, CROP_ECONOMICS_FILENAME), "r", encoding="utf-8") as f:
            economics = json.load(f)
        with open(os.path.join(local_path, HARVEST_INDEX_FILENAME), "r", encoding="utf-8") as f:
            harvest_indices = json.load(f)
        df = pd.read_csv(os.path.join(local_path, DATASET_FILENAME))
    except FileNotFoundError:
        # No compatible user artifacts under this run — fall back to fixed data.
        import shutil  # pylint: disable=import-outside-toplevel
        shutil.rmtree(tmp, ignore_errors=True)
        return None

    logger.info("Downloaded user reference data from MLflow run_id=%s", run_id)
    return economics, harvest_indices, df, tmp
