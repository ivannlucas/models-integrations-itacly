import logging

from app.application.dto.stats_dto import StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import ModelNotLoadedError

logger = logging.getLogger(__name__)


class ModelRuntimeService:
    def __init__(self, plugin: ModelPluginPort) -> None:
        self._plugin = plugin

    def stats(self) -> StatsResponse:
        return self._plugin.stats()

    def is_loaded(self) -> bool:
        return self._plugin.is_loaded()
