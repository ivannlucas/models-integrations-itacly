"""Use case layer for retrieving model statistics."""
import logging
from typing import Callable

from app.application.dto.base_version_dto import VersionInfo
from app.application.dto.stats_dto import StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort


logger = logging.getLogger(__name__)


class GetStatsUseCase:
    """Use case for retrieving model statistics."""
    def __init__(
        self, plugin: ModelPluginPort, base_version_provider: Callable[[], dict] | None = None,
    ) -> None:
        """Initialize the use case with a model plugin.

        base_version_provider is set only for models that version their base (it fills
        StatsResponse.base_version); None leaves the plugin's stats untouched.
        """
        self._plugin = plugin
        self._base_version_provider = base_version_provider

    def execute(self, mlflow_run_id: str = "") -> StatsResponse:
        """Execute the use case and return model statistics from the plugin."""
        if mlflow_run_id:
            logger.info("Retrieving user-trained model stats from MLflow run_id=%s", mlflow_run_id)
        response = self._plugin.stats(mlflow_run_id=mlflow_run_id)
        if self._base_version_provider is not None:
            response = response.model_copy(
                update={"base_version": VersionInfo(**self._base_version_provider())}
            )
        return response
