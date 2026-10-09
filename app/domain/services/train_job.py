"""State of an asynchronous training job (job_id = the MLflow run id the platform pre-creates)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TrainJobStatus(str, Enum):
    """Lifecycle: queued -> running -> succeeded | failed."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in (TrainJobStatus.SUCCEEDED, TrainJobStatus.FAILED)


@dataclass(frozen=True)
class TrainJobRecord:
    """One training job. Times are epoch seconds; ``updated_at`` doubles as the heartbeat."""

    job_id: str
    model_id: str
    status: TrainJobStatus
    submitted_at: float
    updated_at: float
    result: dict | None = None
    error: str | None = None
    error_type: str | None = None
