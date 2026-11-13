"""Test doubles for async batch prediction (importable by name so `spawn` children unpickle them)."""
from __future__ import annotations

import os
import time
from dataclasses import replace
from typing import Annotated, Literal, Union

import numpy as np
from pydantic import Field

from app.application.dto.plugin_dto import BatchRequestBase, BatchResponseBase, PluginDTO
from app.domain.ports.predict_executor_port import PredictExecutorPort
from app.domain.ports.predict_job_store_port import PredictJobStorePort
from app.domain.services.exceptions import (
    ArchiveLimitExceededError,
    InvalidVideoError,
    UserModelUnavailableError,
)
from app.domain.services.predict_job import PredictJobRecord
from tests.unit.train_job_fakes import FileJobStore


class InMemoryPredictJobStore(PredictJobStorePort):
    """Single-process store; ``infos`` keeps what create() received."""

    def __init__(self) -> None:
        self.records: dict[str, PredictJobRecord] = {}
        self.infos: dict[str, dict] = {}

    def get(self, job_id: str) -> PredictJobRecord | None:
        return self.records.get(job_id)

    def create(self, record: PredictJobRecord, info: dict[str, str]) -> None:
        self.infos[record.job_id] = info
        self.records[record.job_id] = record

    def save(self, record: PredictJobRecord) -> None:
        self.records[record.job_id] = record

    def heartbeat(self, job_id: str, at: float) -> None:
        record = self.records.get(job_id)
        if record is not None:
            self.records[job_id] = replace(record, updated_at=at)


class FilePredictJobStore(FileJobStore, PredictJobStorePort):
    """Picklable, cross-process store (one JSON file per job)."""

    def create(self, record: PredictJobRecord, info: dict[str, str]) -> None:
        self.save(record)


class RecordingPredictExecutor(PredictExecutorPort):
    """Records submissions without running anything; ``busy`` is set by the test."""

    def __init__(self, busy: bool = False, fail_on_submit: Exception | None = None) -> None:
        self.busy = busy
        self.fail_on_submit = fail_on_submit
        self.submitted: list[dict] = []

    def is_busy(self) -> bool:
        return self.busy

    def submit(self, *, job_id, plugin_class, request, store, extra_predict_exceptions=()):
        if self.fail_on_submit is not None:
            raise self.fail_on_submit
        self.submitted.append({
            "job_id": job_id, "plugin_class": plugin_class, "request": request,
            "extra_predict_exceptions": extra_predict_exceptions,
        })


class InlinePredictExecutor(PredictExecutorPort):
    """Runs the job synchronously in this process (for router tests)."""

    def __init__(self) -> None:
        self.busy = False

    def is_busy(self) -> bool:
        return self.busy

    def submit(self, *, job_id, plugin_class, request, store, extra_predict_exceptions=()):
        # pylint: disable=import-outside-toplevel  # avoids a cycle at collection time
        from app.infrastructure.predict_jobs.process_executor import run_predict_job
        run_predict_job(
            job_id=job_id, plugin_class=plugin_class, request=request, store=store,
            extra_predict_exceptions=extra_predict_exceptions, heartbeat_s=3600,
        )


# ── fake plugin with the real batch/inline DTO shapes ─────────────────────────

class FakeBatchRequest(BatchRequestBase):
    """Batch request like ml5's (data_path + user model run)."""
    mlflow_run_id: str = ""


class FakeInlineRequest(PluginDTO):
    """Inline request."""
    mode: Literal["inline"] = "inline"
    x: float = 0.0


class FakeBatchResponse(BatchResponseBase):
    """Batch response."""


class FakeInlineResponse(PluginDTO):
    """Inline response."""
    model_id: str
    prediction: float


FakePredictRequest = Annotated[
    Union[FakeBatchRequest, FakeInlineRequest], Field(discriminator="mode")
]
FakePredictResponse = Union[FakeBatchResponse, FakeInlineResponse]

# data_path -> exception raised by EchoPredictPlugin, with the status the synchronous /predict
# answers (InvalidVideoError is in the model's extra_predict_exceptions).
PREDICT_ERRORS = {
    "archive": (ArchiveLimitExceededError("zip demasiado grande"), 413),
    "video": (InvalidVideoError("Cannot open video file"), 422),
    "user-model": (UserModelUnavailableError("run sin modelo"), 422),
    "boom": (RuntimeError("bug"), 500),
}
EXTRA_PREDICT_EXCEPTIONS = (InvalidVideoError,)


class EchoPredictPlugin:
    """predict_batch requires load(); returns numpy values (the use case makes them JSON-safe)."""

    def __init__(self) -> None:
        self.loaded = False

    def load(self) -> None:
        """Marks the plugin as loaded."""
        self.loaded = True

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> FakeBatchResponse:
        """Echo ``data_path``, the user model run and the pid; raise PREDICT_ERRORS[data_path]."""
        if not self.loaded:
            raise RuntimeError("predict_batch() called before load()")
        if data_path in PREDICT_ERRORS:
            raise PREDICT_ERRORS[data_path][0]
        if data_path.startswith("sleep:"):
            time.sleep(float(data_path.split(":", 1)[1]))
        return FakeBatchResponse(
            model_id="fake-model",
            predictions=[
                {"frame": np.int64(i), "score": np.float32(0.5), "run": mlflow_run_id,
                 "pid": os.getpid()}
                for i in range(3)
            ],
        )

    def predict_inline(self, *, features, model_key=None, threshold=None, mlflow_run_id=""):
        """Inline prediction: doubles x."""
        del model_key, threshold, mlflow_run_id
        return FakeInlineResponse(model_id="fake-model", prediction=2 * features["x"])


class SlowPredictPlugin(EchoPredictPlugin):
    """Never finishes on its own (for the watchdog)."""

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> FakeBatchResponse:
        time.sleep(3600)
        raise AssertionError("unreachable")


def batch_body(data_path: str = "s3://b/video.mp4", mlflow_run_id: str = "") -> dict:
    """JSON body of a batch /predict."""
    return {"mode": "batch", "data_path": data_path, "mlflow_run_id": mlflow_run_id}
