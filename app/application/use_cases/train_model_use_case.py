from app.domain.ports.model_plugin_port import ModelPluginPort


class TrainModelUseCase:
    def __init__(self, plugin: ModelPluginPort) -> None:
        self._plugin = plugin

    def execute(self, *, data_path: str) -> dict:
        return self._plugin.train(data_path=data_path)
