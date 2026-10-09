"""Port for persisting asynchronous training job state."""
from abc import ABC, abstractmethod

from app.domain.services.train_job import TrainJobRecord


class TrainJobStorePort(ABC):
    """Where job state lives. Must be shareable with (picklable for) the training child process."""

    @abstractmethod
    def get(self, job_id: str) -> TrainJobRecord | None:
        """Return the job, or None if no job exists for this id."""

    @abstractmethod
    def save(self, record: TrainJobRecord) -> None:
        """Create or overwrite the job."""

    @abstractmethod
    def heartbeat(self, job_id: str, at: float) -> None:
        """Bump ``updated_at`` only (never touches status)."""
