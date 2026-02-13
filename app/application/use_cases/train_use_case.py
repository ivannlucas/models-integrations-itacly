from app.domain.ports.ml_plugin_port import MLPluginPort, TrainResult

class TrainUseCase:
    def __init__(self, plugin: MLPluginPort) -> None:
        self.plugin = plugin

    def execute(self, data_path: str) -> TrainResult:
        return self.plugin.train(data_path=data_path)