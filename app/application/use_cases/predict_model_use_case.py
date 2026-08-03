"""Generic predict use case for model plugins."""
import inspect
import logging
from typing import Any

from pydantic import BaseModel

from app.domain.ports.model_plugin_port import ModelPluginPort

logger = logging.getLogger(__name__)


class PredictModelUseCase:
    """Generic predict use case.

    Works with any plugin that implements ModelPluginPort. The plugin returns
    its own typed response model (batch or inline), so this use case only routes
    the request to the right plugin method and returns the result unchanged.
    """

    def __init__(self, plugin: ModelPluginPort) -> None:
        """Initialize the use case with a model plugin."""
        self._plugin = plugin

    def execute(self, request: Any) -> BaseModel:
        """Execute the prediction, routing to batch or inline mode based on the request."""
        mlflow_run_id = getattr(request, "mlflow_run_id", "")
        if request.mode == "batch":
            logger.info("Executing batch prediction, data_path=%s, mlflow_run_id=%s", request.data_path, mlflow_run_id or "(standard)")
            batch_kwargs: dict[str, Any] = {"data_path": request.data_path, "mlflow_run_id": mlflow_run_id}
            # Only ml34's predict_batch declares model_key (GA-vs-MLP dispatch, mirrors
            # predict_inline below) — every other plugin's predict_batch is
            # (*, data_path, mlflow_run_id) with no **kwargs, so pass it conditionally.
            if "model_key" in inspect.signature(self._plugin.predict_batch).parameters:
                batch_kwargs["model_key"] = getattr(request, "model_key", None)
            return self._plugin.predict_batch(**batch_kwargs)
        logger.info("Executing inline prediction, mlflow_run_id=%s", mlflow_run_id or "(standard)")
        features = request.model_dump(exclude={"mode", "model_key", "threshold", "mlflow_run_id", "data_path"})
        kwargs: dict[str, Any] = {
            "features": features,
            "model_key": getattr(request, "model_key", None),
            "threshold": getattr(request, "threshold", None),
            "mlflow_run_id": mlflow_run_id,
        }
        # Only m47_dnsl_fallas_maquinaria_pasteurizado's predict_inline declares data_path —
        # every other plugin's signature is (*, features, model_key, threshold, mlflow_run_id)
        # with no **kwargs, so passing data_path unconditionally raised TypeError for them.
        if "data_path" in inspect.signature(self._plugin.predict_inline).parameters:
            kwargs["data_path"] = getattr(request, "data_path", None)
        return self._plugin.predict_inline(**kwargs)
