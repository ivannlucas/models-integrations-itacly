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
        logger.info("Executing training, data_path=%s", getattr(request, "data_path", ""))
        data_path = getattr(request, "data_path", "")
        user_id = getattr(request, "user_id", "")
        model_id = getattr(request, "model_id", "")
        return self._plugin.train(data_path=data_path, user_id=user_id, model_id=model_id)
