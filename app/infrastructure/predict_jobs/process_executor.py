"""Run an async batch prediction in a separate process (multiprocessing ``spawn``).

Same mechanics as the async training (``app.infrastructure.train_jobs.process_executor``): the
child builds its own plugin (``plugin_class()`` + ``load()``), heart-beats into the store, runs at
a lower CPU priority and a watchdog in the parent stops it after ``PREDICT_JOB_MAX_DURATION_S``.
The child runs the same ``PredictModelUseCase`` as the synchronous /predict, so the stored result
is the body the synchronous endpoint would have answered.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import threading
import time
from dataclasses import replace
from typing import Any

from app.application.use_cases.predict_model_use_case import PredictModelUseCase
from app.domain.ports.predict_executor_port import PredictExecutorPort
from app.domain.ports.predict_job_store_port import PredictJobStorePort
from app.domain.services.predict_job import (
    TIMEOUT_ERROR_TYPE,
    PredictJobStatus,
    max_duration_error,
    predict_error_http_status,
)
from app.infrastructure.train_jobs.process_executor import (
    _heartbeat_loop,
    _lower_priority,
    _stop,
    _warn_if_multiple_workers,
)

logger = logging.getLogger(__name__)

HEARTBEAT_S = float(os.getenv("PREDICT_JOB_HEARTBEAT_S", "60"))
MAX_DURATION_S = float(os.getenv("PREDICT_JOB_MAX_DURATION_S", "43200"))
NICE = int(os.getenv("PREDICT_JOB_NICE", "10"))
MAX_CONCURRENT = int(os.getenv("PREDICT_JOB_MAX_CONCURRENT", "1"))


def run_predict_job(  # pylint: disable=too-many-arguments
    *,
    job_id: str,
    plugin_class: type,
    request: Any,
    store: PredictJobStorePort,
    extra_predict_exceptions: tuple[type[BaseException], ...] = (),
    heartbeat_s: float = HEARTBEAT_S,
) -> None:
    """Child-process entry point (also callable in-process).

    Prediction errors go to ``store`` with the HTTP status the synchronous /predict would have
    answered; store errors raise. A job already terminal is never overwritten (monotonic status).
    """
    logging.basicConfig(level=logging.INFO)
    record = store.get(job_id)
    if record is None:
        logger.error("Prediction job %s not found in the store; aborting", job_id)
        return
    if record.status.is_terminal:
        logger.warning(
            "Prediction job %s is already %s; not predicting", job_id, record.status.value
        )
        return
    record = replace(record, status=PredictJobStatus.RUNNING, updated_at=time.time())
    store.save(record)

    stop = threading.Event()
    beater = threading.Thread(
        target=_heartbeat_loop, args=(store, job_id, stop, heartbeat_s), daemon=True
    )
    beater.start()
    try:
        plugin = plugin_class()
        plugin.load()
        response = PredictModelUseCase(plugin).execute(request)
        # The JSON body FastAPI sends for the synchronous /predict (by_alias, json mode).
        result = response.model_dump(mode="json", by_alias=True)
        final = replace(
            record, status=PredictJobStatus.SUCCEEDED, result=result, updated_at=time.time()
        )
        logger.info("Prediction job %s succeeded", job_id)
    except Exception as exc:  # pylint: disable=broad-exception-caught  # reported as a failed job
        logger.exception("Prediction job %s failed", job_id)
        final = replace(
            record, status=PredictJobStatus.FAILED, updated_at=time.time(),
            error=str(exc) or type(exc).__name__, error_type=type(exc).__name__,
            http_status=predict_error_http_status(exc, extra_predict_exceptions),
        )
    finally:
        # Join before the final save so a late heartbeat can never overwrite the terminal state.
        stop.set()
        beater.join()
    current = store.get(job_id)
    if current is not None and current.status.is_terminal:
        logger.warning(
            "Prediction job %s finished as %s but was already reported %s (%s); "
            "keeping the reported state",
            job_id, final.status.value, current.status.value, current.error_type,
        )
        return
    store.save(final)


def _child_main(*, nice: int, **job: Any) -> None:
    """Process target: only the child is reniced (run_predict_job is also called in-process)."""
    _lower_priority(nice)
    run_predict_job(**job)


def _watch(proc: Any, job_id: str, store: PredictJobStorePort, max_duration_s: float) -> None:
    """Stop the child once it exceeds ``max_duration_s`` and report the job PredictJobTimeout."""
    proc.join(max_duration_s)
    if not proc.is_alive():
        return
    logger.error(
        "Prediction job %s exceeded %.0f s; stopping pid=%s", job_id, max_duration_s, proc.pid
    )
    _stop(proc)  # frees the concurrency slot even if the store is down
    try:
        current = store.get(job_id)
        if current is None or current.status.is_terminal:
            return
        store.save(replace(
            current, status=PredictJobStatus.FAILED, updated_at=time.time(),
            error_type=TIMEOUT_ERROR_TYPE, error=max_duration_error(max_duration_s),
            http_status=500,
        ))
    except Exception:  # pylint: disable=broad-exception-caught  # watchdog must never crash
        logger.exception("Could not mark prediction job %s as timed out", job_id)


class ProcessPredictExecutor(PredictExecutorPort):
    """One child process per job; busy while ``max_concurrent`` children are alive."""

    def __init__(
        self,
        heartbeat_s: float = HEARTBEAT_S,
        max_duration_s: float = MAX_DURATION_S,
        nice: int = NICE,
        max_concurrent: int = MAX_CONCURRENT,
    ) -> None:
        self._ctx = mp.get_context("spawn")  # fork + CUDA/torch threads deadlock or fail
        self._heartbeat_s = heartbeat_s
        self._max_duration_s = max_duration_s
        self._nice = nice
        self._max_concurrent = max(1, max_concurrent)
        self._procs: dict[str, Any] = {}
        self._lock = threading.Lock()
        _warn_if_multiple_workers()

    def is_busy(self) -> bool:
        with self._lock:
            # is_alive() also reaps finished children.
            self._procs = {job_id: p for job_id, p in self._procs.items() if p.is_alive()}
            return len(self._procs) >= self._max_concurrent

    def submit(  # pylint: disable=too-many-arguments
        self,
        *,
        job_id: str,
        plugin_class: type,
        request: Any,
        store: PredictJobStorePort,
        extra_predict_exceptions: tuple[type[BaseException], ...] = (),
    ) -> None:
        proc = self._ctx.Process(
            target=_child_main,
            kwargs={
                "job_id": job_id, "plugin_class": plugin_class, "request": request,
                "store": store, "extra_predict_exceptions": tuple(extra_predict_exceptions),
                "heartbeat_s": self._heartbeat_s, "nice": self._nice,
            },
            name=f"predict-{job_id}",
            daemon=False,  # daemonic processes cannot have children (e.g. torch DataLoader workers)
        )
        proc.start()
        with self._lock:
            self._procs[job_id] = proc
        threading.Thread(
            target=_watch, args=(proc, job_id, store, self._max_duration_s),
            name=f"predict-watch-{job_id}", daemon=True,
        ).start()
        logger.info("Prediction job %s started in child process pid=%s", job_id, proc.pid)
