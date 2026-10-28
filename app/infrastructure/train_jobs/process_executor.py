"""Run plugin training in a separate process (multiprocessing ``spawn``).

The child builds its own plugin (``plugin_class()`` + ``load()``) so training never shares the GIL,
memory or in-memory model with the process serving predictions, and is discarded when it ends.
It runs at a lower CPU priority (``TRAIN_JOB_NICE``) so a saturated CPU starves training, not the
liveness probes, and a watchdog thread in the parent stops it after ``TRAIN_JOB_MAX_DURATION_S``.
"""
from __future__ import annotations

import functools
import json
import logging
import multiprocessing as mp
import os
import threading
import time
from dataclasses import replace
from typing import Any

from app.domain.ports.train_executor_port import TrainExecutorPort
from app.domain.ports.train_job_store_port import TrainJobStorePort
from app.domain.services.train_job import (
    TIMEOUT_ERROR_TYPE,
    TrainJobStatus,
    max_duration_error,
    train_error_http_status,
)

logger = logging.getLogger(__name__)

HEARTBEAT_S = float(os.getenv("TRAIN_JOB_HEARTBEAT_S", "60"))
MAX_DURATION_S = float(os.getenv("TRAIN_JOB_MAX_DURATION_S", "43200"))
NICE = int(os.getenv("TRAIN_JOB_NICE", "10"))
_KILL_GRACE_S = 10.0  # between SIGTERM and SIGKILL


@functools.cache  # once per process, not once per model
def _warn_if_multiple_workers() -> None:
    """The 409 lock and the idempotency check live in this process: they only hold with 1 worker."""
    workers = os.getenv("WEB_CONCURRENCY", "").strip()  # uvicorn/gunicorn --workers default
    if workers.isdigit() and int(workers) > 1:
        logger.warning(
            "WEB_CONCURRENCY=%s: async training only guarantees one training per model (409) and "
            "idempotent job_ids within a single process; run the service with 1 worker and 1 replica",
            workers,
        )


def _json_default(value: Any) -> Any:
    # numpy scalars/arrays (tolist/item) and datetimes inside free-form metric dicts.
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _to_jsonable(response: Any) -> dict:
    if hasattr(response, "model_dump"):
        data = response.model_dump()
    elif isinstance(response, dict):
        data = response
    else:
        raise TypeError(f"train() returned {type(response).__name__}; expected a pydantic model or a dict")
    return json.loads(json.dumps(data, default=_json_default))


def _heartbeat_loop(store: TrainJobStorePort, job_id: str, stop: threading.Event, every_s: float) -> None:
    while not stop.wait(every_s):
        try:
            store.heartbeat(job_id, time.time())
        except Exception:
            logger.warning("Heartbeat failed for training job %s", job_id, exc_info=True)


def run_train_job(
    *,
    job_id: str,
    plugin_class: type,
    train_kwargs: dict,
    store: TrainJobStorePort,
    heartbeat_s: float = HEARTBEAT_S,
) -> None:
    """Child-process entry point (also callable in-process). Training errors go to ``store``; store errors raise.

    A job already terminal (e.g. marked lost by GET while this child was queued or stuck) is never
    overwritten: job status is monotonic, so consumers can trust the first terminal state they see.
    """
    logging.basicConfig(level=logging.INFO)
    record = store.get(job_id)
    if record is None:
        logger.error("Training job %s not found in the store; aborting", job_id)
        return
    if record.status.is_terminal:
        logger.warning("Training job %s is already %s; not training", job_id, record.status.value)
        return
    record = replace(record, status=TrainJobStatus.RUNNING, updated_at=time.time())
    store.save(record)

    stop = threading.Event()
    beater = threading.Thread(target=_heartbeat_loop, args=(store, job_id, stop, heartbeat_s), daemon=True)
    beater.start()
    try:
        plugin = plugin_class()
        plugin.load()
        result = _to_jsonable(plugin.train(**train_kwargs))
        final = replace(record, status=TrainJobStatus.SUCCEEDED, result=result, updated_at=time.time())
        logger.info("Training job %s succeeded", job_id)
    except Exception as exc:
        logger.exception("Training job %s failed", job_id)
        final = replace(
            record, status=TrainJobStatus.FAILED, updated_at=time.time(),
            error=str(exc) or type(exc).__name__, error_type=type(exc).__name__,
            http_status=train_error_http_status(exc),
        )
    finally:
        # Join before the final save so a late heartbeat can never overwrite the terminal state.
        stop.set()
        beater.join()
    current = store.get(job_id)
    if current is not None and current.status.is_terminal:
        logger.warning(
            "Training job %s finished as %s but was already reported %s (%s); keeping the reported state",
            job_id, final.status.value, current.status.value, current.error_type,
        )
        return
    store.save(final)


def _lower_priority(nice: int) -> None:
    try:
        os.nice(nice)
    except OSError:
        logger.warning("Could not lower the training process priority (nice %s)", nice, exc_info=True)


def _child_main(*, nice: int, **job: Any) -> None:
    """Process target: only the child is reniced (run_train_job is also called in-process)."""
    _lower_priority(nice)
    run_train_job(**job)


def _stop(proc: Any) -> None:
    proc.terminate()
    proc.join(_KILL_GRACE_S)
    if proc.is_alive():
        proc.kill()
        proc.join()


def _watch(proc: Any, job_id: str, store: TrainJobStorePort, max_duration_s: float) -> None:
    """Stop the child once it exceeds ``max_duration_s`` and report the job failed (TrainJobTimeout).

    A hung train() keeps heart-beating, so the stale check alone would leave it running forever.
    """
    proc.join(max_duration_s)
    if not proc.is_alive():
        return
    logger.error("Training job %s exceeded %.0f s; stopping pid=%s", job_id, max_duration_s, proc.pid)
    _stop(proc)  # frees the 409 slot even if the store is down
    try:
        current = store.get(job_id)
        if current is None or current.status.is_terminal:
            return
        store.save(replace(
            current, status=TrainJobStatus.FAILED, updated_at=time.time(), error_type=TIMEOUT_ERROR_TYPE,
            error=max_duration_error(max_duration_s), http_status=500,
        ))
    except Exception:  # pylint: disable=broad-exception-caught
        logger.exception("Could not mark training job %s as timed out", job_id)


class ProcessTrainExecutor(TrainExecutorPort):
    """One child process per job; ``is_busy`` while any submitted child is alive."""

    def __init__(
        self, heartbeat_s: float = HEARTBEAT_S, max_duration_s: float = MAX_DURATION_S, nice: int = NICE,
    ) -> None:
        self._ctx = mp.get_context("spawn")  # fork + torch/OpenMP threads can deadlock
        self._heartbeat_s = heartbeat_s
        self._max_duration_s = max_duration_s
        self._nice = nice
        self._procs: dict[str, Any] = {}
        self._lock = threading.Lock()
        _warn_if_multiple_workers()

    def is_busy(self) -> bool:
        with self._lock:
            # is_alive() also reaps finished children.
            self._procs = {job_id: p for job_id, p in self._procs.items() if p.is_alive()}
            return bool(self._procs)

    def submit(self, *, job_id: str, plugin_class: type, train_kwargs: dict, store: TrainJobStorePort) -> None:
        proc = self._ctx.Process(
            target=_child_main,
            kwargs={
                "job_id": job_id, "plugin_class": plugin_class, "train_kwargs": train_kwargs,
                "store": store, "heartbeat_s": self._heartbeat_s, "nice": self._nice,
            },
            name=f"train-{job_id}",
            daemon=False,  # daemonic processes cannot have children (e.g. torch DataLoader workers)
        )
        proc.start()
        with self._lock:
            self._procs[job_id] = proc
        threading.Thread(
            target=_watch, args=(proc, job_id, store, self._max_duration_s),
            name=f"train-watch-{job_id}", daemon=True,
        ).start()
        logger.info("Training job %s started in child process pid=%s", job_id, proc.pid)
