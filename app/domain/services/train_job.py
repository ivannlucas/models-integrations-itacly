"""State of an asynchronous training job (job_id = the MLflow run id the platform pre-creates)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.domain.services.exceptions import (
    ArchiveLimitExceededError,
    ModelPersistenceError,
    TrainingNotSupportedError,
)

LOST_ERROR_TYPE = "TrainJobLost"
TIMEOUT_ERROR_TYPE = "TrainJobTimeout"


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
    """One training job. Times are epoch seconds; ``updated_at`` doubles as the heartbeat.

    ``http_status`` (failed only) is the code the synchronous /train would have answered.
    """

    job_id: str
    model_id: str
    status: TrainJobStatus
    submitted_at: float
    updated_at: float
    result: dict | None = None
    error: str | None = None
    error_type: str | None = None
    http_status: int | None = None


def train_error_http_status(exc: BaseException) -> int:
    """HTTP status of a training error, shared by the synchronous /train and the async job.

    isinstance in the same order as the synchronous handler, so subclasses (pandas ParserError,
    UnicodeDecodeError, plugin input errors… all ValueError) map like their base class.
    """
    if isinstance(exc, TrainingNotSupportedError):
        return 501
    if isinstance(exc, ArchiveLimitExceededError):  # a ValueError: must go before it
        return 413
    if isinstance(exc, (FileNotFoundError, ValueError)):
        return 400
    if isinstance(exc, ModelPersistenceError):
        return 502
    return 500  # BaseReloadError and anything unexpected


def max_duration_error(max_duration_s: float) -> str:
    """User-facing message of a job stopped for exceeding TRAIN_JOB_MAX_DURATION_S."""
    limit = f"{max_duration_s / 3600:g} h" if max_duration_s >= 3600 else f"{max_duration_s:g} s"
    return f"El entrenamiento superó la duración máxima ({limit}) y se ha detenido."
