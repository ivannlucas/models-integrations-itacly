"""MLflow helpers for ml6: download a user-retrained ensemble and upload a new one.

A retrained ensemble lives ONLY in its MLflow run, under ``model/``:
``<name>.pt`` (state_dict of each base model of the task) + ``metadata.json``
(``{"task", "class_names", "model_names"}``). The base artifacts in
artifacts/ml6_meat_poultry_disease_classification/ are never written.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile

from app.domain.services.mlflow_tracker import BaseMLflowTracker, require_user_model
from app.plugins.ml6_meat_poultry_disease_classification.constants import (
    ARTIFACT_FOLDER_NAME,
    CLASS_NAMES,
    MLFLOW_ARTIFACT_PATH,
    MLFLOW_METADATA_FILENAME,
    MODEL_FILENAMES,
    TASK_MODELS,
)

logger = logging.getLogger(__name__)


@require_user_model
def download_user_model_from_mlflow(run_id: str):
    """Return ``(task, {name: model}, device, temp_dir)`` for the ensemble saved in ``run_id``.

    ``None`` (→ UserModelUnavailableError, 422) when the run has no complete ensemble.
    Caller MUST ``shutil.rmtree(temp_dir)`` in a ``finally``.
    """
    tmp = tempfile.mkdtemp(prefix="mlflow_ml6_poultry_")
    try:
        tracker = BaseMLflowTracker(run_id)
        local_path = tracker.download_artifacts(tmp, artifact_path=MLFLOW_ARTIFACT_PATH)
        metadata_path = os.path.join(local_path, MLFLOW_METADATA_FILENAME) if local_path else ""
        if not local_path or not os.path.exists(metadata_path):
            logger.warning("MLflow run_id=%s sin %s en '%s'", run_id, MLFLOW_METADATA_FILENAME,
                           MLFLOW_ARTIFACT_PATH)
            shutil.rmtree(tmp, ignore_errors=True)
            return None
        with open(metadata_path, encoding="utf-8") as fh:
            metadata = json.load(fh)
        task = metadata["task"]
        if task not in TASK_MODELS or tuple(metadata["class_names"]) != CLASS_NAMES[task]:
            raise ValueError(f"metadata.json no corresponde a un ensemble de ml6: {metadata}")
        missing = [n for n in TASK_MODELS[task]
                   if not os.path.exists(os.path.join(local_path, MODEL_FILENAMES[n]))]
        if missing:
            logger.warning("MLflow run_id=%s: faltan los modelos %s del ensemble %s",
                           run_id, missing, task)
            shutil.rmtree(tmp, ignore_errors=True)
            return None

        # Imported lazily: the module stays importable (hygiene tests) without building models.
        # pylint: disable-next=import-outside-toplevel
        from app.plugins.ml6_meat_poultry_disease_classification.model_loader import (
            build_from_checkpoint,
            safe_device,
        )

        device = safe_device()
        models = {
            name: build_from_checkpoint(
                name, task, os.path.join(local_path, MODEL_FILENAMES[name]), device,
            )
            for name in TASK_MODELS[task]
        }
        logger.info("Ensemble %s del usuario descargado de MLflow run_id=%s", task, run_id)
        return task, models, device, tmp
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def upload_artifacts_to_mlflow(
    artifact_dir: str, mlflow_run_id: str, metrics: dict | None = None,
) -> None:
    """Upload ``artifact_dir`` as the run's ``model/`` and log the final metrics."""
    tracker = BaseMLflowTracker(mlflow_run_id)
    tracker.connect(mlflow_run_id)
    if metrics:
        tracker.log_metrics(metrics)
        tracker.set_tags({"model_id": ARTIFACT_FOLDER_NAME})
    tracker.upload_artifacts(artifact_dir, artifact_path=MLFLOW_ARTIFACT_PATH)
