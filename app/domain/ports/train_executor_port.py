"""Port for running a training job in the background."""
from abc import ABC, abstractmethod

from app.domain.ports.train_job_store_port import TrainJobStorePort


class TrainExecutorPort(ABC):
    """Runs ``plugin_class().train(**train_kwargs)`` away from the request, reporting into ``store``."""

    @abstractmethod
    def is_busy(self) -> bool:
        """True while a previously submitted job is still running."""

    @abstractmethod
    def submit(self, *, job_id: str, plugin_class: type, train_kwargs: dict, store: TrainJobStorePort) -> None:
        """Start the job and return immediately."""
