"""
Dependency injection container for model plugins.
Define aquí el ModelContainer, que se encarga de instanciar y cargar el plugin concreto,
y de proporcionar los casos de uso y el servicio runtime asociados.
"""
import logging

from app.application.use_cases.get_stats_use_case import GetStatsUseCase
from app.application.use_cases.predict_model_use_case import PredictModelUseCase
from app.application.use_cases.train_model_use_case import TrainModelUseCase
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.model_runtime_service import ModelRuntimeService

logger = logging.getLogger(__name__)


class ModelContainer:
    """Generic DI container for any model plugin.

    Wires up use cases and the runtime service around a concrete ModelPluginPort.
    The batch/inline response classes are passed in so each model keeps its own
    typed Pydantic response schema.
    """

    def __init__(
        self,
        plugin: ModelPluginPort,
        batch_response_cls: type,
        inline_response_cls: type,
    ) -> None:
        """Recibe el plugin concreto y las clases de respuesta, y crea los casos de uso."""
        self._plugin = plugin
        self._service = ModelRuntimeService(plugin)
        self.predict_use_case = PredictModelUseCase(plugin, batch_response_cls, inline_response_cls)
        self.stats_use_case = GetStatsUseCase(plugin)
        self.train_use_case = TrainModelUseCase(plugin)

    def init(self) -> None:
        """Carga el plugin (si no se ha cargado ya)."""
        logger.info("Initializing container — loading plugin %s ...", type(self._plugin).__name__)
        self._plugin.load()
        logger.info("Plugin %s loaded successfully.", type(self._plugin).__name__)

    @property
    def service(self) -> ModelRuntimeService:
        """Devuelve el servicio runtime asociado al plugin."""
        return self._service
