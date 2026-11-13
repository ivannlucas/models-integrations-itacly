"""Async batch prediction job state, one MLflow run per job.

A prediction has no MLflow run of its own (the ``mlflow_run_id`` of the request is the user's
retrained model and must not collect one artifact per prediction), so each job gets a run in the
``PREDICT_JOB_EXPERIMENT`` experiment, named after the job and found by its
``async_predict.job_id`` tag. Status and timestamps are tags ``async_predict.*`` (any replica reads
them, they survive restarts); the PredictResponse is the gzip artifact
``async_predict/result.json.gz`` (a long video gives tens of MB of JSON, ~5x smaller gzipped).
"""
from __future__ import annotations

import gzip
import json
import os
import tempfile
import threading

from app.domain.ports.predict_job_store_port import PredictJobStorePort
from app.domain.services.exceptions import (
    TrainJobResultUnreadableError,
    TrainJobRunNotFoundError,
)
from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.domain.services.predict_job import PredictJobRecord, PredictJobStatus
from app.infrastructure.train_jobs.mlflow_job_store import _mlflow_errors

TAG_PREFIX = "async_predict."
JOB_ID_TAG = f"{TAG_PREFIX}job_id"
RESULT_DIR = "async_predict"
RESULT_FILE = "result.json.gz"
RESULT_ARTIFACT = f"{RESULT_DIR}/{RESULT_FILE}"
DEFAULT_EXPERIMENT = "datagia-async-predict"
_MAX_TAG_CHARS = 4000  # well under MLflow's tag value limit
_ALREADY_EXISTS = "RESOURCE_ALREADY_EXISTS"


class MLflowPredictJobStore(PredictJobStorePort):
    """PredictJobStorePort backed by one MLflow run per job (tags + one gzip JSON artifact)."""

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
        client=None,
    ) -> None:
        self._tracking_uri = tracking_uri or BaseMLflowTracker.TRACKING_URI
        self._experiment_name = experiment_name or os.getenv(
            "PREDICT_JOB_EXPERIMENT", DEFAULT_EXPERIMENT
        )
        self._client = client
        self._experiment_id: str | None = None
        self._runs: dict[str, str] = {}  # job_id -> run_id (a job's run never changes)
        self._lock = threading.Lock()

    def __getstate__(self) -> dict:
        # The prediction child process builds its own client and lock.
        state = self.__dict__.copy()
        state["_client"] = None
        state["_lock"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    @property
    def client(self):
        """MLflow client, built lazily on first use."""
        if self._client is None:
            from mlflow.tracking import MlflowClient  # pylint: disable=import-outside-toplevel
            self._client = MlflowClient(tracking_uri=self._tracking_uri)
        return self._client

    # ── run lookup ────────────────────────────────────────────────────────────

    def _get_experiment_id(self, create: bool) -> str | None:
        if self._experiment_id is not None:
            return self._experiment_id
        experiment = self.client.get_experiment_by_name(self._experiment_name)
        if experiment is None and create:
            from mlflow.exceptions import MlflowException  # pylint: disable=import-outside-toplevel
            try:
                self._experiment_id = self.client.create_experiment(self._experiment_name)
                return self._experiment_id
            except MlflowException as exc:  # another replica created it first
                if getattr(exc, "error_code", "") != _ALREADY_EXISTS:
                    raise
            experiment = self.client.get_experiment_by_name(self._experiment_name)
        if experiment is None:
            return None
        self._experiment_id = experiment.experiment_id
        return self._experiment_id

    def _find_run_id(self, job_id: str) -> str | None:
        """Run holding the job, or None. ``job_id`` is validated upstream (safe in the filter)."""
        if job_id in self._runs:
            return self._runs[job_id]
        experiment_id = self._get_experiment_id(create=False)
        if experiment_id is None:
            return None
        runs = self.client.search_runs(
            [experiment_id], filter_string=f"tags.`{JOB_ID_TAG}` = '{job_id}'", max_results=1,
        )
        if not runs:
            return None
        self._runs[job_id] = runs[0].info.run_id
        return self._runs[job_id]

    # ── PredictJobStorePort ───────────────────────────────────────────────────

    def get(self, job_id: str) -> PredictJobRecord | None:
        try:
            with _mlflow_errors(job_id):
                run_id = self._find_run_id(job_id)
                if run_id is None:
                    return None
                run = self.client.get_run(run_id)
        except TrainJobRunNotFoundError:  # run deleted between search and get
            self._runs.pop(job_id, None)
            return None
        tags = run.data.tags
        status = tags.get(f"{TAG_PREFIX}status")
        if not status:
            return None
        status = PredictJobStatus(status)
        return PredictJobRecord(
            job_id=job_id,
            model_id=tags.get(f"{TAG_PREFIX}model_id", ""),
            status=status,
            submitted_at=float(tags[f"{TAG_PREFIX}submitted_at"]),
            updated_at=float(tags[f"{TAG_PREFIX}updated_at"]),
            result=(
                self._load_result(job_id, run_id) if status is PredictJobStatus.SUCCEEDED else None
            ),
            error=tags.get(f"{TAG_PREFIX}error") or None,
            error_type=tags.get(f"{TAG_PREFIX}error_type") or None,
            http_status=(
                int(tags[f"{TAG_PREFIX}http_status"])
                if tags.get(f"{TAG_PREFIX}http_status") else None
            ),
        )

    def create(self, record: PredictJobRecord, info: dict[str, str]) -> None:
        with self._lock, _mlflow_errors(record.job_id):
            experiment_id = self._get_experiment_id(create=True)
            tags = {JOB_ID_TAG: record.job_id, f"{TAG_PREFIX}model_id": record.model_id}
            tags.update({
                f"{TAG_PREFIX}{key}": str(value)[:_MAX_TAG_CHARS]
                for key, value in info.items() if value
            })
            run = self.client.create_run(experiment_id, tags=tags, run_name=record.job_id)
            self._runs[record.job_id] = run.info.run_id
            self._save(record)

    def save(self, record: PredictJobRecord) -> None:
        with _mlflow_errors(record.job_id):
            self._save(record)

    def _run_id(self, job_id: str) -> str:
        run_id = self._find_run_id(job_id)
        if run_id is None:
            raise TrainJobRunNotFoundError(f"No hay run de MLflow para la predicción {job_id}")
        return run_id

    def _save(self, record: PredictJobRecord) -> None:
        run_id = self._run_id(record.job_id)
        if record.result is not None:
            self._log_result(run_id, record.result)
        tags = {
            "model_id": record.model_id,
            "submitted_at": str(record.submitted_at),
            "updated_at": str(record.updated_at),
        }
        if record.error:
            tags["error"] = record.error[:_MAX_TAG_CHARS]
        if record.error_type:
            tags["error_type"] = record.error_type
        if record.http_status is not None:
            tags["http_status"] = str(record.http_status)
        for key, value in tags.items():
            self.client.set_tag(run_id, f"{TAG_PREFIX}{key}", value)
        # Status last: a reader never sees "succeeded" before the result exists.
        self.client.set_tag(run_id, f"{TAG_PREFIX}status", record.status.value)
        if record.status.is_terminal:
            self.client.set_terminated(
                run_id, "FINISHED" if record.status is PredictJobStatus.SUCCEEDED else "FAILED"
            )

    def _log_result(self, run_id: str, result: dict) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, RESULT_FILE)
            with gzip.open(path, "wt", encoding="utf-8") as fh:
                json.dump(result, fh, ensure_ascii=False)
            self.client.log_artifact(run_id, path, RESULT_DIR)

    def heartbeat(self, job_id: str, at: float) -> None:
        with _mlflow_errors(job_id):
            self.client.set_tag(self._run_id(job_id), f"{TAG_PREFIX}updated_at", str(at))

    def _load_result(self, job_id: str, run_id: str) -> dict:
        # Download the folder, not the file: through the platform's S3 proxy, downloading the lone
        # file gives an empty directory (see MLflowTrainJobStore._load_result).
        prefix = f"La predicción {job_id} terminó (succeeded) pero"
        with tempfile.TemporaryDirectory() as tmp:
            try:
                with _mlflow_errors(job_id):
                    local = os.path.join(
                        self.client.download_artifacts(run_id, RESULT_DIR, tmp), RESULT_FILE
                    )
            except TrainJobRunNotFoundError as exc:
                raise TrainJobResultUnreadableError(
                    f"{prefix} MLflow no tiene {RESULT_ARTIFACT}: {exc}"
                ) from exc
            except OSError as exc:  # network errors already became TrainJobStoreUnavailableError
                raise TrainJobResultUnreadableError(
                    f"{prefix} no se pudo descargar {RESULT_ARTIFACT}: {type(exc).__name__}: {exc}"
                ) from exc
            if not os.path.isfile(local):
                found = "una carpeta" if os.path.isdir(local) else "nada"
                raise TrainJobResultUnreadableError(
                    f"{prefix} al descargar {RESULT_ARTIFACT} de MLflow se obtiene {found} "
                    "en vez del fichero"
                )
            try:
                with gzip.open(local, "rt", encoding="utf-8") as fh:
                    return json.load(fh)
            except (OSError, ValueError, EOFError) as exc:
                raise TrainJobResultUnreadableError(
                    f"{prefix} {RESULT_ARTIFACT} no se puede leer: {exc}"
                ) from exc
