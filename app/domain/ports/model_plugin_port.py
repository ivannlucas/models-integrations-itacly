"""Abstract port interface for model plugins."""
from abc import ABC, abstractmethod

from pydantic import BaseModel

from app.application.dto.stats_dto import StatsResponse


class ModelPluginPort(ABC):
    """Interface that all model plugins must implement."""

    @abstractmethod
    def load(self) -> None:
        """Load model artifacts from disk."""

    def load_for_training(self, **train_kwargs) -> None:
        """Load what ``train(**train_kwargs)`` needs in the async training child process.

        The child runs next to the serving process in the same pod, so whatever it loads is a
        second copy on top of the served model. By default it loads everything (``load()``);
        a plugin whose ``train()`` uses only part of its artifacts (or none) overrides this so
        the pod does not hold the model twice.
        """
        del train_kwargs
        self.load()

    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True if the model is ready for inference."""

    @abstractmethod
    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> BaseModel:
        """Run batch inference and return the model's typed batch response."""

    @abstractmethod
    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> BaseModel:
        """Run inline inference on a single feature dict and return the typed inline response."""

    @abstractmethod
    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Return model metadata and runtime statistics."""

    @abstractmethod
    def train(self, *, data_path: str, mlflow_run_id: str) -> BaseModel:
        """Train the model with the provided data and return the typed train response."""
