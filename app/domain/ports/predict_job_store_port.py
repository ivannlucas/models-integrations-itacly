"""Port for persisting asynchronous batch prediction job state."""
from abc import ABC, abstractmethod

from app.domain.services.predict_job import PredictJobRecord


class PredictJobStorePort(ABC):
    """Where job state lives.

    Must be shareable with (picklable for) the prediction child process.
    """

    @abstractmethod
    def get(self, job_id: str) -> PredictJobRecord | None:
        """Return the job, or None if no job exists for this id."""

    @abstractmethod
    def create(self, record: PredictJobRecord, info: dict[str, str]) -> None:
        """Register a new job; ``info`` is descriptive metadata (data_path, user model run…)."""

    @abstractmethod
    def save(self, record: PredictJobRecord) -> None:
        """Overwrite an existing job."""

    @abstractmethod
    def heartbeat(self, job_id: str, at: float) -> None:
        """Bump ``updated_at`` only (never touches status)."""
