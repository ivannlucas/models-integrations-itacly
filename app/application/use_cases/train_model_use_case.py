"""Generic train use case for model plugins."""
import inspect
from typing import Any
from app.domain.ports.model_plugin_port import ModelPluginPort
import logging

logger = logging.getLogger(__name__)


class TrainModelUseCase:
    """Generic train use case."""

    def __init__(self, plugin: ModelPluginPort) -> None:
        """Initialize the use case with a model plugin."""
        self._plugin = plugin

    def execute(self, request: Any) -> dict:
        """Executes the training process."""
        data_path = getattr(request, "data_path", "")
        mlflow_run_id = getattr(request, "mlflow_run_id", "")
        logger.info("Executing training, data_path=%s, mlflow_run_id=%s", data_path, mlflow_run_id)
        kwargs: dict[str, Any] = {"data_path": data_path, "mlflow_run_id": mlflow_run_id}
        # Plugin-specific train options (ml40 "system", ml47 "mode"…): forward every request
        # field that the plugin's train() declares; anything absent keeps the plugin default.
        for name in inspect.signature(self._plugin.train).parameters:
            if name not in kwargs and hasattr(request, name):
                kwargs[name] = getattr(request, name)
        return self._plugin.train(**kwargs)
