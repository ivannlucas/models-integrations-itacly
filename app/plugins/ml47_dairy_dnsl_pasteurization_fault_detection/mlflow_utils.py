from __future__ import annotations

import logging
import shutil
from pathlib import Path

from app.domain.services.mlflow_tracker import BaseMLflowTracker, require_user_model
from app.plugins.ml47_dairy_dnsl_pasteurization_fault_detection.constants import ARTIFACT_FOLDER_NAME

logger = logging.getLogger(__name__)


@require_user_model
def download_user_model_from_mlflow(run_id: str):
    import tempfile
    tmp = tempfile.mkdtemp(prefix="mlflow_m47_")
    try:
        local_path = BaseMLflowTracker(run_id).download_artifacts(tmp, artifact_path="model")
        if not local_path:
            shutil.rmtree(tmp, ignore_errors=True)
            return None

        from app.plugins.ml47_dairy_dnsl_pasteurization_fault_detection.model_loader import load_artifacts_from_dir

        # Same loader as the base model: same dropout, weights_only=True on the user's state_dict.
        model, scaler, feature_cols, ts1_mean_train = load_artifacts_from_dir(Path(local_path))

        logger.info("Downloaded user model from MLflow run_id=%s", run_id)
        return model, scaler, feature_cols, ts1_mean_train, tmp
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


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
        tracker.set_tags({"model_id": "ml47-dairy-dnsl-pasteurization-fault-detection"})

    tracker.upload_artifacts(artifact_dir, artifact_path="model")

    logger.info("Artifacts uploaded to MLflow (uri=%s) run_id=%s", uri, mlflow_run_id)
    return mlflow_run_id
