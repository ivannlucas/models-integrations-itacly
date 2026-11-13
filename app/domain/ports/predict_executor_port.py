"""Port for running a batch prediction job in the background."""
from abc import ABC, abstractmethod
from typing import Any

from app.domain.ports.predict_job_store_port import PredictJobStorePort


class PredictExecutorPort(ABC):
    """Runs the generic predict use case on ``plugin_class()`` away from the request.

    Progress and the outcome are reported into ``store``.
    """

    @abstractmethod
    def is_busy(self) -> bool:
        """True while no further job can start (the concurrency limit is reached)."""

    @abstractmethod
    def submit(  # pylint: disable=too-many-arguments
        self,
        *,
        job_id: str,
        plugin_class: type,
        request: Any,
        store: PredictJobStorePort,
        extra_predict_exceptions: tuple[type[BaseException], ...] = (),
    ) -> None:
        """Start the job and return immediately."""
