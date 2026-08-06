from __future__ import annotations

import logging
import os

import joblib
import torch

from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.plugins.m47_dnsl_fallas_maquinaria_pasteurizado.constants import (
    ARTIFACT_FOLDER_NAME,
    FEATURE_COLUMNS_FILENAME,
    MODEL_FILENAME,
    SCALER_FILENAME,
    TS1_MEAN_FILENAME,
)

logger = logging.getLogger(__name__)


def download_user_model_from_mlflow(run_id: str):
    import tempfile
    tmp = tempfile.mkdtemp(prefix="mlflow_m47_")
    local_path = BaseMLflowTracker(run_id).download_artifacts(tmp, artifact_path="model")
    if not local_path:
        return None

    from app.plugins.m47_dnsl_fallas_maquinaria_pasteurizado.model_loader import CNN_Pasteurizer

    state_dict_path = os.path.join(local_path, MODEL_FILENAME)
    scaler_path = os.path.join(local_path, SCALER_FILENAME)
    feature_cols_path = os.path.join(local_path, FEATURE_COLUMNS_FILENAME)
    ts1_path = os.path.join(local_path, TS1_MEAN_FILENAME)

    scaler = joblib.load(scaler_path)
    feature_cols = joblib.load(feature_cols_path)
    ts1_mean_train = joblib.load(ts1_path)

    n_sensors = len(feature_cols)
    model = CNN_Pasteurizer(n_sensors=n_sensors, n_classes=3, dropout_prob=0.5)
    model.load_state_dict(torch.load(state_dict_path, map_location="cpu", weights_only=False))
    model.eval()

    logger.info("Downloaded user model from MLflow run_id=%s", run_id)
    return model, scaler, feature_cols, ts1_mean_train, tmp


def upload_artifacts_to_mlflow(
    artifact_dir: str,
    mlflow_run_id: str = "",
    metrics: dict | None = None,
) -> str:
    """Upload training artifacts to MLflow and return the run_id.

    If mlflow_run_id is provided, logs to that existing run.
    Otherwise starts a new run under the m47 experiment.
    """
    return _upload_with_uri(BaseMLflowTracker.TRACKING_URI, artifact_dir, mlflow_run_id, metrics)


def _upload_with_uri(
    uri: str,
    artifact_dir: str,
    mlflow_run_id: str = "",
    metrics: dict | None = None,
) -> str:
    import mlflow

    def make_tracker(run_id: str) -> BaseMLflowTracker:
        t = BaseMLflowTracker(run_id)
        t.TRACKING_URI = uri
        t.connect(run_id)
        return t

    mlflow.set_tracking_uri(uri)

    if mlflow_run_id:
        tracker = make_tracker(mlflow_run_id)
    else:
        mlflow.set_experiment(ARTIFACT_FOLDER_NAME)
        with mlflow.start_run() as run:
            mlflow_run_id = run.info.run_id
            tracker = make_tracker(mlflow_run_id)

    if metrics:
        tracker.log_metrics(metrics)
        tracker.set_tags({"model_id": "m47-dnsl-fallas-maquinaria-pasteurizado"})

    tracker.upload_artifacts(artifact_dir, artifact_path="model")

    logger.info("Artifacts uploaded to MLflow (uri=%s) run_id=%s", uri, mlflow_run_id)
    return mlflow_run_id
