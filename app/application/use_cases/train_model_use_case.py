"""Generic train use case for model plugins."""
import inspect
from typing import Any
from app.domain.ports.model_plugin_port import ModelPluginPort
import logging

logger = logging.getLogger(__name__)


def build_train_kwargs(plugin: Any, request: Any) -> dict[str, Any]:
    """Map a TrainRequest to ``plugin.train`` kwargs (shared by the sync and the async paths)."""
    kwargs: dict[str, Any] = {
        "data_path": getattr(request, "data_path", ""),
        "mlflow_run_id": getattr(request, "mlflow_run_id", ""),
    }
    # Only ml40's train declares system (refrigeracion/aireado chosen by the user).
    if "system" in inspect.signature(plugin.train).parameters:
        kwargs["system"] = getattr(request, "system", None)
    return kwargs


class TrainModelUseCase:
    """Generic train use case."""

    def __init__(self, plugin: ModelPluginPort) -> None:
        """Initialize the use case with a model plugin."""
        self._plugin = plugin

    def execute(self, request: Any) -> dict:
        """Executes the training process."""
        kwargs = build_train_kwargs(self._plugin, request)
        logger.info("Executing training, data_path=%s, mlflow_run_id=%s", kwargs["data_path"], kwargs["mlflow_run_id"])
        return self._plugin.train(**kwargs)
