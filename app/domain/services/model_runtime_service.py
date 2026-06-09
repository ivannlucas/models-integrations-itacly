"""Service layer for interacting with model plugins."""
import logging

from app.application.dto.stats_dto import StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort

logger = logging.getLogger(__name__)


class ModelRuntimeService:
    """Service layer that interacts with model plugins for stats and health checks."""

    def __init__(self, plugin: ModelPluginPort) -> None:
        """Initialize the service with a model plugin."""
        self._plugin = plugin

    def stats(self) -> StatsResponse:
        """Fetches model statistics from the plugin."""
        return self._plugin.stats()

    def is_loaded(self) -> bool:
        """Checks if the model is loaded."""
        return self._plugin.is_loaded()
