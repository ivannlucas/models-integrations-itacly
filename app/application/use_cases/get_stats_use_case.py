"""Use case layer for retrieving model statistics."""
import logging
from app.application.dto.stats_dto import StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort


logger = logging.getLogger(__name__)


class GetStatsUseCase:
    """Use case for retrieving model statistics."""
    def __init__(self, plugin: ModelPluginPort) -> None:
        """Initialize the use case with a model plugin."""
        self._plugin = plugin

    def execute(self, mlflow_run_id: str = "") -> StatsResponse:
        """Execute the use case and return model statistics from the plugin."""
        if mlflow_run_id:
            logger.info("Retrieving user-trained model stats from MLflow run_id=%s", mlflow_run_id)
        return self._plugin.stats(mlflow_run_id=mlflow_run_id)
