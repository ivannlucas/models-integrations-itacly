from app.application.dto.stats_dto import StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort


class GetStatsUseCase:
    """Use case that retrieves statistics and metadata from a model plugin."""

    def __init__(self, plugin: ModelPluginPort) -> None:
        """Store *plugin* for use during execution."""
        self._plugin = plugin

    def execute(self) -> StatsResponse:
        """Execute the use case and return the plugin's stats response."""
        return self._plugin.stats()
