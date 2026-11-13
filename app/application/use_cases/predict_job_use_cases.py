"""Async batch prediction use cases: POST /predict?wait=false and GET /predict/jobs/{job_id}."""
from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import replace
from typing import Any, Callable

from app.domain.ports.predict_executor_port import PredictExecutorPort
from app.domain.ports.predict_job_store_port import PredictJobStorePort
from app.domain.services.exceptions import PredictJobBusyError
from app.domain.services.predict_job import (
    LOST_ERROR_TYPE,
    TIMEOUT_ERROR_TYPE,
    PredictJobRecord,
    PredictJobStatus,
    lost_error,
    max_duration_error,
    predict_error_http_status,
    validate_job_id,
)

logger = logging.getLogger(__name__)


class SubmitPredictJobUseCase:  # pylint: disable=too-few-public-methods
    """Register a batch prediction job (idempotent by job_id) and start it in the background."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        plugin: Any,
        model_id: str,
        store: PredictJobStorePort,
        executor: PredictExecutorPort,
        *,
        extra_predict_exceptions: tuple[type[BaseException], ...] = (),
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._plugin = plugin
        self._model_id = model_id
        self._store = store
        self._executor = executor
        self._extra = tuple(extra_predict_exceptions)
        self._clock = clock
        # FastAPI runs sync endpoints in a threadpool: check-then-submit must be atomic.
        self._lock = threading.Lock()

    def execute(self, request: Any, job_id: str = "") -> PredictJobRecord:
        """Queue the batch prediction under ``job_id`` (a new UUID if empty) and return its job.

        ValueError → 400, PredictJobBusyError → 409, TrainJobStoreUnavailableError → 503.
        """
        job_id = validate_job_id(job_id) if (job_id or "").strip() else uuid.uuid4().hex
        with self._lock:
            existing = self._store.get(job_id)
            if existing is not None:
                if existing.model_id != self._model_id:
                    raise ValueError(f"El job_id {job_id} pertenece al modelo {existing.model_id}")
                # Idempotent: a redelivered request (e.g. Celery) never starts a second prediction.
                return existing
            if self._executor.is_busy():
                raise PredictJobBusyError(
                    f"Ya hay una predicción asíncrona en curso del modelo {self._model_id}"
                )
            now = self._clock()
            record = PredictJobRecord(
                job_id=job_id, model_id=self._model_id, status=PredictJobStatus.QUEUED,
                submitted_at=now, updated_at=now,
            )
            self._store.create(record, {
                "data_path": str(getattr(request, "data_path", "") or ""),
                "mlflow_run_id": str(getattr(request, "mlflow_run_id", "") or ""),
            })
            try:
                self._executor.submit(
                    job_id=job_id,
                    plugin_class=type(self._plugin),
                    request=request,
                    store=self._store,
                    extra_predict_exceptions=self._extra,
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught  # recorded as a failed job
                logger.exception(
                    "Could not start prediction job %s for model '%s'", job_id, self._model_id
                )
                failed = replace(
                    record,
                    status=PredictJobStatus.FAILED,
                    updated_at=self._clock(),
                    error=f"No se pudo lanzar el proceso de predicción: {exc}",
                    error_type=type(exc).__name__,
                    http_status=predict_error_http_status(exc, self._extra),
                )
                self._store.save(failed)
                return failed
            logger.info("Prediction job %s queued for model '%s'", job_id, self._model_id)
            return record


class GetPredictJobUseCase:  # pylint: disable=too-few-public-methods
    """Return a job; a non-terminal job that outlived PREDICT_JOB_MAX_DURATION_S (timeout) or
    stopped heart-beating (lost) is reported, and persisted, as failed."""

    def __init__(
        self,
        model_id: str,
        store: PredictJobStorePort,
        clock: Callable[[], float] = time.time,
        stale_after_s: float | None = None,
        max_duration_s: float | None = None,
    ) -> None:
        self._model_id = model_id
        self._store = store
        self._clock = clock
        self._stale_after_s = (
            stale_after_s if stale_after_s is not None
            else float(os.getenv("PREDICT_JOB_STALE_AFTER_S", "600"))
        )
        self._max_duration_s = (
            max_duration_s if max_duration_s is not None
            else float(os.getenv("PREDICT_JOB_MAX_DURATION_S", "43200"))
        )

    def execute(self, job_id: str) -> PredictJobRecord | None:
        """Return the job (None if unknown/other model), marking timed-out or lost jobs failed."""
        try:
            job_id = validate_job_id(job_id)
        except ValueError:
            return None  # cannot exist: POST never accepts such an id
        record = self._store.get(job_id)
        if record is None or record.model_id != self._model_id:
            return None
        if record.status.is_terminal:
            return record
        now = self._clock()
        if now - record.submitted_at > self._max_duration_s:
            # The pod that runs it stops the child at the same deadline (ProcessPredictExecutor).
            record = replace(
                record, status=PredictJobStatus.FAILED, updated_at=now,
                error_type=TIMEOUT_ERROR_TYPE, error=max_duration_error(self._max_duration_s),
                http_status=500,
            )
        elif now - record.updated_at > self._stale_after_s:
            record = replace(
                record, status=PredictJobStatus.FAILED, updated_at=now,
                error_type=LOST_ERROR_TYPE, error=lost_error(self._stale_after_s), http_status=500,
            )
        else:
            return record
        self._store.save(record)
        logger.warning(
            "Prediction job %s for model '%s' marked as %s", job_id, self._model_id,
            record.error_type,
        )
        return record
