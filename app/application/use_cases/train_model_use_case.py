from typing import Any
from app.domain.ports.model_plugin_port import ModelPluginPort


class TrainModelUseCase:
    def __init__(self, plugin: ModelPluginPort) -> None:
        self._plugin = plugin

    def execute(self, request: Any) -> dict:
        request_data = request.model_dump()

        # Check if 'data_path' is the ONLY key in the dictionary
        if list(request_data.keys()) == ["data_path"]:
            return self._plugin.train(data_path=request_data["data_path"])

        else:
            return self._plugin.train(**request_data, data_path=request.pop("data_path"))
