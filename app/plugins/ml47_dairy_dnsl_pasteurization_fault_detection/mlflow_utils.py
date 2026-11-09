"""MLflow helper de ml47: descarga del modelo reentrenado y subida de artefactos."""
from __future__ import annotations

import logging
from pathlib import Path

from app.domain.services.mlflow_artifacts import download_run_artifacts
from app.domain.services.mlflow_tracker import BaseMLflowTracker, require_user_model
from app.plugins.ml47_dairy_dnsl_pasteurization_fault_detection.constants import (
    ARTIFACT_FOLDER_NAME,
)

logger = logging.getLogger(__name__)


@require_user_model
def download_user_model_from_mlflow(run_id: str):
    """Descarga el modelo reentrenado del run.

    Devuelve (model, scaler, feature_cols, ts1_mean_train, temp_dir); el llamador borra temp_dir.
    """
    loaded = download_run_artifacts(run_id, _load, prefix="mlflow_m47_")
    if loaded is None:
        return None
    bundle, tmp = loaded
    return (*bundle, tmp)


def _load(local_path: str):
    """Rebuild (model, scaler, feature_cols, ts1_mean_train) from a downloaded ``model`` folder."""
    # pylint: disable-next=import-outside-toplevel
    from app.plugins.ml47_dairy_dnsl_pasteurization_fault_detection.model_loader import (
        load_artifacts_from_dir,
    )

    # Same loader as the base model: same dropout, weights_only=True on the user's state_dict.
    return load_artifacts_from_dir(Path(local_path))


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
    import mlflow  # pylint: disable=import-outside-toplevel

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
        tracker.set_tags({"model_id": "ml47-dairy-dnsl-pasteurization-fault-detection"})

    tracker.upload_artifacts(artifact_dir, artifact_path="model")

    logger.info("Artifacts uploaded to MLflow (uri=%s) run_id=%s", uri, mlflow_run_id)
    return mlflow_run_id
