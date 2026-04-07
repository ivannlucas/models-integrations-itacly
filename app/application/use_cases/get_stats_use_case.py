from app.application.dto.stats_dto import StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort


class GetStatsUseCase:
    def __init__(self, plugin: ModelPluginPort) -> None:
        self._plugin = plugin

    def execute(self) -> StatsResponse:
        return self._plugin.stats()
