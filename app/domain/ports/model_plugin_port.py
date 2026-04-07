from abc import ABC, abstractmethod

from app.application.dto.stats_dto import StatsResponse


class ModelPluginPort(ABC):
    @abstractmethod
    def load(self) -> None:
        """Load model artifacts from disk."""

    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True if the model is ready for inference."""

    @abstractmethod
    def predict_batch(self, *, data_path: str) -> dict:
        """Run batch inference on a CSV/image directory and return a predictions dict."""

    @abstractmethod
    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
    ) -> dict:
        """Run inline inference on a single feature dict and return a prediction dict."""

    @abstractmethod
    def stats(self) -> StatsResponse:
        """Return model metadata and runtime statistics."""
