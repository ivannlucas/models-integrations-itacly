"""Async training job state stored on the MLflow run itself (job_id = run_id).

Tags ``async_train.*`` hold status and timestamps (any replica can read them, they survive restarts);
the plugin's TrainResponse goes to the ``async_train/result.json`` artifact of the same run.
"""
from __future__ import annotations

import json
import tempfile
from contextlib import contextmanager

import requests

from app.domain.ports.train_job_store_port import TrainJobStorePort
from app.domain.services.exceptions import TrainJobRunNotFoundError, TrainJobStoreUnavailableError
from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.domain.services.train_job import TrainJobRecord, TrainJobStatus

TAG_PREFIX = "async_train."
RESULT_ARTIFACT = "async_train/result.json"
_MAX_ERROR_CHARS = 4000  # well under MLflow's tag value limit
_NOT_FOUND = "RESOURCE_DOES_NOT_EXIST"


@contextmanager
def _mlflow_errors(job_id: str):
    """Translate MLflow/transport failures into domain errors (missing run → 400, anything else → 503)."""
    from mlflow.exceptions import MlflowException
    try:
        yield
    except MlflowException as exc:
        if getattr(exc, "error_code", "") == _NOT_FOUND:
            raise TrainJobRunNotFoundError(f"El run {job_id} no existe en MLflow") from exc
        raise TrainJobStoreUnavailableError(f"MLflow: {exc}") from exc
    except (requests.exceptions.RequestException, OSError) as exc:
        raise TrainJobStoreUnavailableError(f"MLflow no accesible: {exc}") from exc


class MLflowTrainJobStore(TrainJobStorePort):
    """TrainJobStorePort backed by MLflow tags + one JSON artifact."""

    def __init__(self, tracking_uri: str | None = None, client=None) -> None:
        self._tracking_uri = tracking_uri or BaseMLflowTracker.TRACKING_URI
        self._client = client

    def __getstate__(self) -> dict:
        # The training child process builds its own client.
        state = self.__dict__.copy()
        state["_client"] = None
        return state

    @property
    def client(self):
        if self._client is None:
            from mlflow.tracking import MlflowClient
            self._client = MlflowClient(tracking_uri=self._tracking_uri)
        return self._client

    def get(self, job_id: str) -> TrainJobRecord | None:
        try:
            with _mlflow_errors(job_id):
                run = self.client.get_run(job_id)
        except TrainJobRunNotFoundError:
            return None
        tags = run.data.tags
        status = tags.get(f"{TAG_PREFIX}status")
        if not status:
            return None
        status = TrainJobStatus(status)
        return TrainJobRecord(
            job_id=job_id,
            model_id=tags.get(f"{TAG_PREFIX}model_id", ""),
            status=status,
            submitted_at=float(tags[f"{TAG_PREFIX}submitted_at"]),
            updated_at=float(tags[f"{TAG_PREFIX}updated_at"]),
            result=self._load_result(job_id) if status is TrainJobStatus.SUCCEEDED else None,
            error=tags.get(f"{TAG_PREFIX}error") or None,
            error_type=tags.get(f"{TAG_PREFIX}error_type") or None,
        )

    def save(self, record: TrainJobRecord) -> None:
        with _mlflow_errors(record.job_id):
            self._save(record)

    def _save(self, record: TrainJobRecord) -> None:
        if record.result is not None:
            self.client.log_text(record.job_id, json.dumps(record.result, ensure_ascii=False), RESULT_ARTIFACT)
        tags = {
            "model_id": record.model_id,
            "submitted_at": str(record.submitted_at),
            "updated_at": str(record.updated_at),
        }
        if record.error:
            tags["error"] = record.error[:_MAX_ERROR_CHARS]
        if record.error_type:
            tags["error_type"] = record.error_type
        for key, value in tags.items():
            self.client.set_tag(record.job_id, f"{TAG_PREFIX}{key}", value)
        # Status last: a reader never sees "succeeded" before result.json exists.
        self.client.set_tag(record.job_id, f"{TAG_PREFIX}status", record.status.value)

    def heartbeat(self, job_id: str, at: float) -> None:
        with _mlflow_errors(job_id):
            self.client.set_tag(job_id, f"{TAG_PREFIX}updated_at", str(at))

    def _load_result(self, job_id: str) -> dict:
        with tempfile.TemporaryDirectory() as tmp, _mlflow_errors(job_id):
            local = self.client.download_artifacts(job_id, RESULT_ARTIFACT, tmp)
            with open(local, encoding="utf-8") as fh:
                return json.load(fh)
