"""Generic predict use case for model plugins."""
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
        if request.mode == "batch":
            logger.info("Executing batch prediction, data_path=%s", request.data_path)
            return self._plugin.predict_batch(data_path=request.data_path, model_id=request.model_id, user_id=request.user_id,)
        logger.info("Executing inline prediction")
        features = request.model_dump(exclude={"mode", "model_key", "threshold"})
        return self._plugin.predict_inline(
            features=features,
            model_key=getattr(request, "model_key", None),
            threshold=getattr(request, "threshold", None),
            model_id=getattr(request, "model_id", None),
            user_id=getattr(request, "user_id", None),
        )
