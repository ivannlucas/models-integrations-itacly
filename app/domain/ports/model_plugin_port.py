"""Abstract port interface for model plugins."""
from abc import ABC, abstractmethod

from pydantic import BaseModel

from app.application.dto.stats_dto import StatsResponse


class ModelPluginPort(ABC):
    """Interface that all model plugins must implement."""

    @abstractmethod
    def load(self) -> None:
        """Load model artifacts from disk."""

    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True if the model is ready for inference."""

    @abstractmethod
    def predict_batch(self, *, data_path: str) -> BaseModel:
        """Run batch inference and return the model's typed batch response."""

    @abstractmethod
    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
    ) -> BaseModel:
        """Run inline inference on a single feature dict and return the typed inline response."""

    @abstractmethod
    def stats(self) -> StatsResponse:
        """Return model metadata and runtime statistics."""

    @abstractmethod
    def train(self, *, data_path: str) -> BaseModel:
        """Train the model with the provided data and return the typed train response."""
