"""Test doubles for async training (importable by name so `spawn` children can unpickle them)."""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, replace

import numpy as np
from pydantic import BaseModel

from app.domain.ports.train_executor_port import TrainExecutorPort
from app.domain.ports.train_job_store_port import TrainJobStorePort
from app.domain.services.train_job import TrainJobRecord, TrainJobStatus


class InMemoryJobStore(TrainJobStorePort):
    """Single-process store."""

    def __init__(self) -> None:
        self.records: dict[str, TrainJobRecord] = {}

    def get(self, job_id: str) -> TrainJobRecord | None:
        return self.records.get(job_id)

    def save(self, record: TrainJobRecord) -> None:
        self.records[record.job_id] = record

    def heartbeat(self, job_id: str, at: float) -> None:
        record = self.records.get(job_id)
        if record is not None:
            self.records[job_id] = replace(record, updated_at=at)


class FileJobStore(TrainJobStorePort):
    """Picklable, cross-process store: one JSON file per job under ``root``."""

    def __init__(self, root: str) -> None:
        self.root = root

    def _path(self, job_id: str) -> str:
        return os.path.join(self.root, f"{job_id}.json")

    def get(self, job_id: str) -> TrainJobRecord | None:
        path = self._path(job_id)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        data["status"] = TrainJobStatus(data["status"])
        return TrainJobRecord(**data)

    def save(self, record: TrainJobRecord) -> None:
        data = asdict(record)
        data["status"] = record.status.value
        tmp = self._path(record.job_id) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        os.replace(tmp, self._path(record.job_id))

    def heartbeat(self, job_id: str, at: float) -> None:
        record = self.get(job_id)
        if record is not None:
            self.save(replace(record, updated_at=at))


class RecordingExecutor(TrainExecutorPort):
    """Records submissions without running anything; ``busy`` is set by the test."""

    def __init__(self, busy: bool = False, fail_on_submit: Exception | None = None) -> None:
        self.busy = busy
        self.fail_on_submit = fail_on_submit
        self.submitted: list[dict] = []

    def is_busy(self) -> bool:
        return self.busy

    def submit(self, *, job_id: str, plugin_class: type, train_kwargs: dict, store: TrainJobStorePort) -> None:
        if self.fail_on_submit is not None:
            raise self.fail_on_submit
        self.submitted.append({"job_id": job_id, "plugin_class": plugin_class, "train_kwargs": train_kwargs})


class FakeTrainResponse(BaseModel):
    detail: str
    metrics: dict = {}
    mlflow_run_id: str = ""


class EchoTrainPlugin:
    """Minimal plugin: train() requires load() and reports the pid it ran in."""

    def __init__(self) -> None:
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def train(self, *, data_path: str, mlflow_run_id: str) -> FakeTrainResponse:
        if not self.loaded:
            raise RuntimeError("train() called before load()")
        return FakeTrainResponse(
            detail="ok", metrics={"data_path": data_path, "pid": os.getpid()}, mlflow_run_id=mlflow_run_id,
        )


class FailingTrainPlugin(EchoTrainPlugin):
    def train(self, *, data_path: str, mlflow_run_id: str) -> FakeTrainResponse:
        raise FileNotFoundError(f"no existe {data_path}")


class SlowTrainPlugin(EchoTrainPlugin):
    """Sleeps ``float(data_path)`` seconds before answering."""

    def train(self, *, data_path: str, mlflow_run_id: str) -> FakeTrainResponse:
        time.sleep(float(data_path))
        return super().train(data_path=data_path, mlflow_run_id=mlflow_run_id)


class NumpyMetricsTrainPlugin(EchoTrainPlugin):
    def train(self, *, data_path: str, mlflow_run_id: str) -> FakeTrainResponse:
        return FakeTrainResponse(
            detail="ok",
            metrics={"f1": np.float32(0.5), "n": np.int64(3), "cm": np.array([[1, 0], [0, 2]])},
            mlflow_run_id=mlflow_run_id,
        )


class SystemTrainPlugin(EchoTrainPlugin):
    """Declares ``system`` like ml40 does, to check kwargs forwarding."""

    def train(self, *, data_path: str, mlflow_run_id: str, system: str | None = None) -> FakeTrainResponse:
        return FakeTrainResponse(detail=f"system={system}", mlflow_run_id=mlflow_run_id)


class InlineExecutor(TrainExecutorPort):
    """Runs the job synchronously in the calling thread (HTTP tests)."""

    def __init__(self, busy: bool = False) -> None:
        self.busy = busy
        self.calls = 0

    def is_busy(self) -> bool:
        return self.busy

    def submit(self, *, job_id: str, plugin_class: type, train_kwargs: dict, store: TrainJobStorePort) -> None:
        from app.infrastructure.train_jobs.process_executor import run_train_job

        self.calls += 1
        run_train_job(job_id=job_id, plugin_class=plugin_class, train_kwargs=train_kwargs, store=store, heartbeat_s=3600)
