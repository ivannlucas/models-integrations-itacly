import logging
from typing import Any

from app.domain.ports.model_plugin_port import ModelPluginPort

logger = logging.getLogger(__name__)


class PredictModelUseCase:
    """Generic predict use case.

    Works with any plugin that implements ModelPluginPort.
    The concrete batch/inline response classes are injected at construction time
    so that each model keeps its own typed response schema.
    """

    def __init__(
        self,
        plugin: ModelPluginPort,
        batch_response_cls: type,
        inline_response_cls: type,
    ) -> None:
        self._plugin = plugin
        self._batch_cls = batch_response_cls
        self._inline_cls = inline_response_cls

    def execute(self, request: Any) -> Any:
        if request.mode == "batch":
            logger.info("Executing batch prediction, data_path=%s", request.data_path)
            result = self._plugin.predict_batch(data_path=request.data_path)
            return self._batch_cls(**result)
        else:
            logger.info("Executing inline prediction")
            features = request.model_dump(exclude={"mode", "model_key", "threshold"})
            result = self._plugin.predict_inline(
                features=features,
                model_key=getattr(request, "model_key", None),
                threshold=getattr(request, "threshold", None),
            )
            return self._inline_cls(**result)
