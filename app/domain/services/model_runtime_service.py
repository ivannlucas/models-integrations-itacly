import logging

from app.application.dto.stats_dto import StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort

logger = logging.getLogger(__name__)


class ModelRuntimeService:
    """Runtime wrapper that delegates service-level operations to a plugin."""

    def __init__(self, plugin: ModelPluginPort) -> None:
        """Wrap *plugin* so use cases can call it through a uniform interface."""
        self._plugin = plugin

    def stats(self) -> StatsResponse:
        """Return the plugin's model statistics and runtime metadata."""
        return self._plugin.stats()

    def is_loaded(self) -> bool:
        """Return True if the underlying plugin has loaded its artifacts."""
        return self._plugin.is_loaded()
