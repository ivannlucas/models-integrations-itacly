"""Async training use cases: POST /train?wait=false and GET /train/{job_id}."""
from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from typing import Any, Callable, Iterator

from app.application.use_cases.train_model_use_case import build_train_kwargs
from app.domain.ports.train_executor_port import TrainExecutorPort
from app.domain.ports.train_job_store_port import TrainJobStorePort
from app.domain.services.exceptions import TrainJobBusyError
from app.domain.services.train_job import (
    LOST_ERROR_TYPE,
    TIMEOUT_ERROR_TYPE,
    TrainJobRecord,
    TrainJobStatus,
    max_duration_error,
    train_error_http_status,
)

logger = logging.getLogger(__name__)


class SubmitTrainJobUseCase:
    """Register a job (job_id = mlflow_run_id) and start it in the background.

    Also the per-model training gate: an async job and a synchronous /train (``sync_training``)
    never run at the same time in this process.
    """

    def __init__(
        self,
        plugin: Any,
        model_id: str,
        store: TrainJobStorePort,
        executor: TrainExecutorPort,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._plugin = plugin
        self._model_id = model_id
        self._store = store
        self._executor = executor
        self._clock = clock
        # FastAPI runs sync endpoints in a threadpool: check-then-submit must be atomic.
        self._lock = threading.Lock()
        self._sync_running = 0

    @contextmanager
    def sync_training(self) -> Iterator[None]:
        """Hold the model for a synchronous /train; TrainJobBusyError if an async job is alive."""
        with self._lock:
            if self._executor.is_busy():
                raise TrainJobBusyError(
                    f"Ya hay un entrenamiento asíncrono en curso del modelo {self._model_id}"
                )
            self._sync_running += 1
        try:
            yield
        finally:
            with self._lock:
                self._sync_running -= 1

    def execute(self, request: Any) -> TrainJobRecord:
        job_id = (getattr(request, "mlflow_run_id", "") or "").strip()
        if not job_id:
            raise ValueError("mlflow_run_id es obligatorio en el entrenamiento asíncrono: es el job_id")
        with self._lock:
            existing = self._store.get(job_id)
            if existing is not None:
                if existing.model_id != self._model_id:
                    raise ValueError(f"El job_id {job_id} pertenece al modelo {existing.model_id}")
                # Idempotent: a redelivered request (e.g. Celery) never starts a second training.
                return existing
            if self._sync_running or self._executor.is_busy():
                raise TrainJobBusyError(f"Ya hay un entrenamiento en curso del modelo {self._model_id}")
            now = self._clock()
            record = TrainJobRecord(
                job_id=job_id, model_id=self._model_id, status=TrainJobStatus.QUEUED,
                submitted_at=now, updated_at=now,
            )
            self._store.save(record)
            try:
                self._executor.submit(
                    job_id=job_id,
                    plugin_class=type(self._plugin),
                    train_kwargs=build_train_kwargs(self._plugin, request),
                    store=self._store,
                )
            except Exception as exc:
                logger.exception("Could not start training job %s for model '%s'", job_id, self._model_id)
                failed = replace(
                    record, status=TrainJobStatus.FAILED, updated_at=self._clock(),
                    error=f"No se pudo lanzar el proceso de entrenamiento: {exc}", error_type=type(exc).__name__,
                    http_status=train_error_http_status(exc),
                )
                self._store.save(failed)
                return failed
            logger.info("Training job %s queued for model '%s'", job_id, self._model_id)
            return record


class GetTrainJobUseCase:
    """Return a job; a non-terminal job that outlived TRAIN_JOB_MAX_DURATION_S (timeout) or stopped
    heart-beating (lost) is reported, and persisted, as failed."""

    def __init__(
        self,
        model_id: str,
        store: TrainJobStorePort,
        clock: Callable[[], float] = time.time,
        stale_after_s: float | None = None,
        max_duration_s: float | None = None,
    ) -> None:
        self._model_id = model_id
        self._store = store
        self._clock = clock
        self._stale_after_s = (
            stale_after_s if stale_after_s is not None else float(os.getenv("TRAIN_JOB_STALE_AFTER_S", "600"))
        )
        self._max_duration_s = (
            max_duration_s if max_duration_s is not None
            else float(os.getenv("TRAIN_JOB_MAX_DURATION_S", "43200"))
        )

    def execute(self, job_id: str) -> TrainJobRecord | None:
        record = self._store.get(job_id)
        if record is None or record.model_id != self._model_id:
            return None
        if record.status.is_terminal:
            return record
        now = self._clock()
        if now - record.submitted_at > self._max_duration_s:
            # The pod that runs it stops the child at the same deadline (ProcessTrainExecutor).
            record = replace(
                record, status=TrainJobStatus.FAILED, updated_at=now, error_type=TIMEOUT_ERROR_TYPE,
                error=max_duration_error(self._max_duration_s), http_status=500,
            )
        elif now - record.updated_at > self._stale_after_s:
            record = replace(
                record, status=TrainJobStatus.FAILED, updated_at=now, error_type=LOST_ERROR_TYPE, http_status=500,
                error=(
                    f"El proceso de entrenamiento dejó de dar señales hace más de {int(self._stale_after_s)} s "
                    "(reinicio del pod o proceso terminado por el sistema)."
                ),
            )
        else:
            return record
        self._store.save(record)
        logger.warning("Training job %s for model '%s' marked as %s", job_id, self._model_id, record.error_type)
        return record
