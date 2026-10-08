"""MLflow helper for ml26 — download a user fine-tuned GRU-PSO model from MLflow."""

from __future__ import annotations

import logging
import shutil
import tempfile

from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.plugins.ml26_wine_sulfite_gru_pso_forecast.constants import MLFLOW_ARTIFACT_PATH
from app.plugins.ml26_wine_sulfite_gru_pso_forecast.model_loader import LoadedModel, load_user_model

logger = logging.getLogger(__name__)


def download_user_model_from_mlflow(run_id: str) -> tuple[LoadedModel, str] | None:
    """Download a user fine-tuned model (model_state.pt + bundle_meta.json) from MLflow.

    Returns (model, temp_dir) or None if the run has no artifacts.
    Caller MUST shutil.rmtree(temp_dir) after inference — use try/finally.
    """
    tmp = tempfile.mkdtemp(prefix="mlflow_ml26_")
    local_path = BaseMLflowTracker(run_id).download_artifacts(
        tmp, artifact_path=MLFLOW_ARTIFACT_PATH
    )
    if not local_path:
        shutil.rmtree(tmp, ignore_errors=True)
        return None
    model = load_user_model(local_path, source=f"mlflow:{run_id}")
    logger.info("Downloaded user model from MLflow run_id=%s", run_id)
    return model, tmp
