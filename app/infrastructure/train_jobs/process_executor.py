"""Run plugin training in a separate process (multiprocessing ``spawn``).

The child builds its own plugin (``plugin_class()`` + ``load()``) so training never shares the GIL,
memory or in-memory model with the process serving predictions, and is discarded when it ends.
"""
from __future__ import annotations

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
from app.domain.services.train_job import TrainJobStatus

logger = logging.getLogger(__name__)

HEARTBEAT_S = float(os.getenv("TRAIN_JOB_HEARTBEAT_S", "60"))


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


class ProcessTrainExecutor(TrainExecutorPort):
    """One child process per job; ``is_busy`` while any submitted child is alive."""

    def __init__(self, heartbeat_s: float = HEARTBEAT_S) -> None:
        self._ctx = mp.get_context("spawn")  # fork + torch/OpenMP threads can deadlock
        self._heartbeat_s = heartbeat_s
        self._procs: dict[str, Any] = {}
        self._lock = threading.Lock()

    def is_busy(self) -> bool:
        with self._lock:
            # is_alive() also reaps finished children.
            self._procs = {job_id: p for job_id, p in self._procs.items() if p.is_alive()}
            return bool(self._procs)

    def submit(self, *, job_id: str, plugin_class: type, train_kwargs: dict, store: TrainJobStorePort) -> None:
        proc = self._ctx.Process(
            target=run_train_job,
            kwargs={
                "job_id": job_id, "plugin_class": plugin_class, "train_kwargs": train_kwargs,
                "store": store, "heartbeat_s": self._heartbeat_s,
            },
            name=f"train-{job_id}",
            daemon=False,  # daemonic processes cannot have children (e.g. torch DataLoader workers)
        )
        proc.start()
        with self._lock:
            self._procs[job_id] = proc
        logger.info("Training job %s started in child process pid=%s", job_id, proc.pid)
