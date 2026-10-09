"""
Dependency injection container for model plugins.
Define aquí el ModelContainer, que se encarga de instanciar y cargar el plugin concreto,
y de proporcionar los casos de uso y el servicio runtime asociados.
"""
import logging

from app.application.use_cases.get_stats_use_case import GetStatsUseCase
from app.application.use_cases.predict_model_use_case import PredictModelUseCase
from app.application.use_cases.train_job_use_cases import GetTrainJobUseCase, SubmitTrainJobUseCase
from app.application.use_cases.train_model_use_case import TrainModelUseCase
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.model_runtime_service import ModelRuntimeService
from app.infrastructure.train_jobs.mlflow_job_store import MLflowTrainJobStore
from app.infrastructure.train_jobs.process_executor import ProcessTrainExecutor

logger = logging.getLogger(__name__)


class ModelContainer:
    """Generic DI container for any model plugin.

    Wires up use cases and the runtime service around a concrete ModelPluginPort.
    Each plugin returns its own typed Pydantic response models, so no response
    classes need to be injected here.
    """

    def __init__(self, plugin: ModelPluginPort, model_id: str = "") -> None:
        """Recibe el plugin concreto y crea los casos de uso."""
        self._plugin = plugin
        self._service = ModelRuntimeService(plugin)
        self.predict_use_case = PredictModelUseCase(plugin)
        self.stats_use_case = GetStatsUseCase(plugin)
        self.train_use_case = TrainModelUseCase(plugin)
        # Async training (POST /train?wait=false): state on the MLflow run, training in a child process.
        # Neither builds anything heavy here: the MLflow client is created lazily, no process starts yet.
        train_job_store = MLflowTrainJobStore()
        self.submit_train_job_use_case = SubmitTrainJobUseCase(
            plugin, model_id, train_job_store, ProcessTrainExecutor()
        )
        self.get_train_job_use_case = GetTrainJobUseCase(model_id, train_job_store)

    def init(self) -> None:
        """Carga el plugin (si no se ha cargado ya)."""
        logger.info("Initializing container — loading plugin %s ...", type(self._plugin).__name__)
        self._plugin.load()
        logger.info("Plugin %s loaded successfully.", type(self._plugin).__name__)

    @property
    def service(self) -> ModelRuntimeService:
        """Devuelve el servicio runtime asociado al plugin."""
        return self._service
