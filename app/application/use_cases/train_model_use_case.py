"""Generic train use case for model plugins."""
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
        return self._plugin.train(data_path=data_path, mlflow_run_id=mlflow_run_id)
