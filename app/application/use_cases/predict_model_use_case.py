"""Generic predict use case for model plugins."""
import inspect
import logging
from typing import Any

import numpy as np
from pydantic import BaseModel

from app.domain.ports.model_plugin_port import ModelPluginPort

logger = logging.getLogger(__name__)


def _to_jsonable(value: Any) -> Any:
    """Recursively convert numpy scalar/array values into native JSON-serializable types.

    Plugins commonly build their loosely-typed response fields (``dict[str, Any]`` batch
    rows, free-form explanation dicts) straight from pandas/numpy computations. Pydantic only
    coerces values for concretely-typed fields, so a stray numpy.int64/float64/bool_/ndarray
    left in an ``Any`` field passes validation but blows up at JSON-encoding time with
    ``PydanticSerializationError: Unable to serialize unknown type``. Normalizing here, once,
    for every plugin's response protects all of them instead of each plugin having to remember
    to cast (some do, e.g. ml9's own `clean_scalar`; several don't).
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


class PredictModelUseCase:
    """Generic predict use case.

    Works with any plugin that implements ModelPluginPort. The plugin returns
    its own typed response model (batch or inline); this use case routes the
    request to the right plugin method and returns a JSON-safe copy of the result.
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
            result = self._plugin.predict_batch(**batch_kwargs)
        else:
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
            result = self._plugin.predict_inline(**kwargs)

        # by_alias=True: some response models (e.g. ml45) declare fields with `alias=...` and no
        # `populate_by_name`, so they only validate from alias-keyed input — dumping by field name
        # would make model_validate reject its own model's data as "field required".
        return type(result).model_validate(_to_jsonable(result.model_dump(mode="python", by_alias=True)))
