"""
Dependency injection container for model plugins.
Define aquí el ModelContainer, que se encarga de instanciar y cargar el plugin concreto,
y de proporcionar los casos de uso y el servicio runtime asociados.

Los plugins que declaran BASE_ARTIFACT_FOLDER (ml13, 14, 15, 16, 17, 18, 19, 21, 23, 24) publican cada
reentreno como nueva versión del modelo base. Para ellos el contenedor serializa /train y
/rollback con un lock, y tras activar una versión crea una instancia nueva del plugin, la carga
y la sustituye; si no carga, el puntero vuelve a la versión anterior y se sigue sirviendo la
instancia vieja.
"""
import logging
import threading
from typing import Any

from app.application.use_cases.get_stats_use_case import GetStatsUseCase
from app.application.use_cases.predict_model_use_case import PredictModelUseCase
from app.application.use_cases.train_model_use_case import TrainModelUseCase
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import BaseReloadError, NoPreviousBaseVersionError
from app.domain.services.model_runtime_service import ModelRuntimeService
from app.infrastructure import base_model_publisher

logger = logging.getLogger(__name__)


class _PromotingTrainUseCase:
    """/train of a model that promotes its base: train, publish, reload (ModelContainer.train)."""

    def __init__(self, container: "ModelContainer") -> None:
        self._container = container

    def execute(self, request: Any) -> Any:
        """Delegate to the container, which owns the lock and the plugin swap."""
        return self._container.train(request)


class ModelContainer:
    """Generic DI container for any model plugin.

    Wires up use cases and the runtime service around a concrete ModelPluginPort.
    Each plugin returns its own typed Pydantic response models, so no response
    classes need to be injected here.
    """

    def __init__(self, plugin: ModelPluginPort) -> None:
        """Recibe el plugin concreto y crea los casos de uso."""
        folder = getattr(type(plugin), "BASE_ARTIFACT_FOLDER", None)
        self._base_store = base_model_publisher._store_for(folder) if folder else None  # pylint: disable=protected-access
        self._lock = threading.Lock()
        # Version of the instance being served; the global pointer can be ahead of it
        # while train() publishes and before the fresh instance is swapped in.
        self._served_version: str | None = None
        self._wire(plugin)

    def _wire(self, plugin: ModelPluginPort) -> None:
        """(Re)build the service and use cases around *plugin*. Each attribute is replaced
        whole, so a request reads either the old or the new instance, never a mix."""
        self._plugin = plugin
        self._service = ModelRuntimeService(plugin)
        self.predict_use_case = PredictModelUseCase(plugin)
        self.stats_use_case = GetStatsUseCase(
            plugin, base_version_provider=self.base_version if self._base_store else None,
        )
        self.train_use_case = _PromotingTrainUseCase(self) if self._base_store else TrainModelUseCase(plugin)

    @property
    def promotes_base(self) -> bool:
        """True when every /train publishes a new base version (BASE_ARTIFACT_FOLDER)."""
        return self._base_store is not None

    def init(self) -> None:
        """Carga el plugin (si no se ha cargado ya)."""
        logger.info("Initializing container — loading plugin %s ...", type(self._plugin).__name__)
        if self._base_store is not None:
            self._base_store.refresh()
            self._served_version = self._base_store.active_version()
        self._plugin.load()
        logger.info("Plugin %s loaded successfully.", type(self._plugin).__name__)

    def base_version(self) -> dict:
        """version.json of the base version being served (not the global pointer)."""
        return self._base_store.version_info(self._served_version)

    def _restore_pointer(self, previous: str) -> None:
        """Point the base back at *previous*; a failure here is logged, never raised, so it
        cannot mask the error that triggered the restore."""
        try:
            self._base_store.set_active_version(previous)
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("Could not restore the base pointer to %s", previous)

    def _reload_or_restore(self, previous: str) -> None:
        """Load a fresh plugin on the active version and swap it in; on failure, point the base
        back at *previous* and keep serving the current instance."""
        failed = self._base_store.active_version()
        try:
            fresh = type(self._plugin)()
            fresh.load()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._restore_pointer(previous)
            logger.exception("Base version %s of %s does not load; back to %s",
                             failed, type(self._plugin).__name__, previous)
            raise BaseReloadError(
                f"La versión {failed} del modelo base no se ha podido cargar "
                f"({type(exc).__name__}: {exc}); el modelo base vuelve a {previous}."
            ) from exc
        self._served_version = failed
        self._wire(fresh)
        logger.info("Plugin %s now serves base version %s", type(fresh).__name__, failed)

    def train(self, request: Any) -> Any:
        """Train (the plugin publishes the new base) and serve the new version."""
        with self._lock:
            previous = self._served_version or self._base_store.active_version()
            try:
                response = TrainModelUseCase(self._plugin).execute(request)
            except Exception:
                # A train() that failed after publishing must not leave the pointer on a
                # version this pod does not serve.
                # Unconditional: current.json may be written even if the in-process
                # version was not updated.
                self._restore_pointer(previous)
                raise
            self._reload_or_restore(previous)
            return response

    def rollback(self, version: str | None = None) -> dict:
        """Activate *version* (the current one's parent by default) and serve it."""
        with self._lock:
            current = self.base_version()
            target = version or current.get("parent")
            if not target:
                raise NoPreviousBaseVersionError(
                    "El modelo base vigente es v0: no hay una versión anterior a la que volver."
                )
            self._base_store.set_active_version(target)
            self._reload_or_restore(current["version"])
            return {"base_version": self._base_store.version_info(self._served_version), "previous_base_version": current}

    @property
    def service(self) -> ModelRuntimeService:
        """Devuelve el servicio runtime asociado al plugin."""
        return self._service
