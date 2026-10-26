"""Async training job state stored on the MLflow run itself (job_id = run_id).

Tags ``async_train.*`` hold status and timestamps (any replica can read them, they survive restarts);
the plugin's TrainResponse goes to the ``async_train/result.json`` artifact of the same run.
"""
from __future__ import annotations

import json
import os
import socket
import tempfile
from contextlib import contextmanager

import requests

from app.domain.ports.train_job_store_port import TrainJobStorePort
from app.domain.services.exceptions import (
    TrainJobResultUnreadableError,
    TrainJobRunNotFoundError,
    TrainJobStoreUnavailableError,
)
from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.domain.services.train_job import TrainJobRecord, TrainJobStatus

TAG_PREFIX = "async_train."
RESULT_DIR = "async_train"
RESULT_FILE = "result.json"
RESULT_ARTIFACT = f"{RESULT_DIR}/{RESULT_FILE}"
_MAX_ERROR_CHARS = 4000  # well under MLflow's tag value limit
_NOT_FOUND = "RESOURCE_DOES_NOT_EXIST"


@contextmanager
def _mlflow_errors(job_id: str):
    """Translate MLflow/transport failures into domain errors (missing run → 400, MLflow or network → 503).

    Only network errors are "MLflow not reachable": a local file error (e.g. reading a downloaded
    artifact) is not retryable and must not be reported as one.
    """
    from mlflow.exceptions import MlflowException
    try:
        yield
    except MlflowException as exc:
        if getattr(exc, "error_code", "") == _NOT_FOUND:
            raise TrainJobRunNotFoundError(f"El run {job_id} no existe en MLflow") from exc
        raise TrainJobStoreUnavailableError(f"MLflow: {exc}") from exc
    except (requests.exceptions.RequestException, ConnectionError, TimeoutError, socket.gaierror) as exc:
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
            http_status=int(tags[f"{TAG_PREFIX}http_status"]) if tags.get(f"{TAG_PREFIX}http_status") else None,
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
        if record.http_status is not None:
            tags["http_status"] = str(record.http_status)
        for key, value in tags.items():
            self.client.set_tag(record.job_id, f"{TAG_PREFIX}{key}", value)
        # Status last: a reader never sees "succeeded" before result.json exists.
        self.client.set_tag(record.job_id, f"{TAG_PREFIX}status", record.status.value)

    def heartbeat(self, job_id: str, at: float) -> None:
        with _mlflow_errors(job_id):
            self.client.set_tag(job_id, f"{TAG_PREFIX}updated_at", str(at))

    def _load_result(self, job_id: str) -> dict:
        # Download the folder, not the file: the platform's S3 proxy answers the listing of
        # "async_train/result.json/" as if it were a folder, so MLflow "downloads" the lone file as an
        # empty directory (IsADirectoryError). The folder listing returns result.json as a file.
        with tempfile.TemporaryDirectory() as tmp:
            try:
                with _mlflow_errors(job_id):
                    local = os.path.join(self.client.download_artifacts(job_id, RESULT_DIR, tmp), RESULT_FILE)
            except TrainJobRunNotFoundError as exc:  # the run exists (it has job tags): the artifact does not
                raise TrainJobResultUnreadableError(
                    f"El entrenamiento {job_id} terminó (succeeded) pero MLflow no tiene {RESULT_ARTIFACT}: {exc}"
                ) from exc
            except OSError as exc:  # network errors already became TrainJobStoreUnavailableError
                raise TrainJobResultUnreadableError(
                    f"El entrenamiento {job_id} terminó (succeeded) pero no se pudo descargar {RESULT_ARTIFACT}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if not os.path.isfile(local):
                found = "una carpeta" if os.path.isdir(local) else "nada"
                raise TrainJobResultUnreadableError(
                    f"El entrenamiento {job_id} terminó (succeeded) pero al descargar {RESULT_ARTIFACT} "
                    f"de MLflow se obtiene {found} en vez del fichero"
                )
            try:
                with open(local, encoding="utf-8") as fh:
                    return json.load(fh)
            except (OSError, ValueError) as exc:
                raise TrainJobResultUnreadableError(
                    f"El entrenamiento {job_id} terminó (succeeded) pero {RESULT_ARTIFACT} no se puede leer: {exc}"
                ) from exc
