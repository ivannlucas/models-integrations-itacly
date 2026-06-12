"""Generic train use case for model plugins."""
from typing import Any
from app.domain.ports.model_plugin_port import ModelPluginPort


class TrainModelUseCase:
    """Generic train use case."""

    def __init__(self, plugin: ModelPluginPort) -> None:
        """Initialize the use case with a model plugin."""
        self._plugin = plugin

    def execute(self, request: Any) -> dict:
        """Executes the training process."""
        request_data = request.model_dump()

        # Check if 'data_path' is the ONLY key in the dictionary
        if list(request_data.keys()) == ["data_path"]:
            return self._plugin.train(data_path=request_data["data_path"])

        return self._plugin.train(**request_data, data_path=request.pop("data_path"))
