from app.domain.ports.ml_plugin_port import MLPluginPort, StatsResult

class StatsUseCase:
    def __init__(self, plugin: MLPluginPort) -> None:
        self.plugin = plugin

    def execute(self) -> StatsResult:
        return self.plugin.get_stats()