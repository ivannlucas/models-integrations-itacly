"""Use case layer for retrieving model statistics. """
from app.application.dto.stats_dto import StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort


class GetStatsUseCase:
    """Use case for retrieving model statistics."""
    def __init__(self, plugin: ModelPluginPort) -> None:
        """Initialize the use case with a model plugin."""
        self._plugin = plugin

    def execute(self) -> StatsResponse:
        """Execute the use case and return model statistics from the plugin."""
        return self._plugin.stats()
