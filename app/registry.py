"""Model registry.

To add a new model:
1. Create app/plugins/<name>/ implementing ModelPluginPort.
2. Add a ModelEntry below — the router and DI are wired automatically.
"""
from dataclasses import dataclass, field
from typing import Any

from app.domain.services.exceptions import NoValidSimulationPointError

# ── Plugin imports ────────────────────────────────────────────────────────────


from app.plugins.ml25_wine_sulphites.plugin import WineSulphitePlugin
from app.plugins.ml25_wine_sulphites.predict_dto import (
    PredictBatchResponse as WineSO2_BatchResp,
    PredictInlineResponse as WineSO2_InlineResp,
    PredictRequest as WineSO2_Request,
    PredictResponse as WineSO2_Response,
)
from app.plugins.ml25_wine_sulphites.train_dto import (
    TrainRequest as WineSO2_TrainReq,
    TrainResponse as WineSO2_TrainResp,
)


# ── Registry entry dataclass ──────────────────────────────────────────────────

@dataclass
class ModelEntry:
    """Registry entry that binds a model ID to its plugin class and Pydantic DTOs."""

    model_id: str
    prefix: str
    version: str
    plugin_class: type
    predict_request_type: Any
    predict_response_type: Any
    batch_response_class: type
    inline_response_class: type
    extra_predict_exceptions: tuple[type[Exception], ...] = field(default_factory=tuple)
    train_request_type: Any = None
    train_response_type: Any = None


# ── Registry ──────────────────────────────────────────────────────────────────

REGISTRY: list[ModelEntry] = [
    ModelEntry(
        model_id="wine-sulphite",
        prefix="/models/ml25_wine_sulphites",
        version="1.2.0",
        plugin_class=WineSulphitePlugin,
        predict_request_type=WineSO2_Request,
        predict_response_type=WineSO2_Response,
        batch_response_class=WineSO2_BatchResp,
        inline_response_class=WineSO2_InlineResp,
        extra_predict_exceptions=(NoValidSimulationPointError,),
        train_request_type=WineSO2_TrainReq,
        train_response_type=WineSO2_TrainResp,
    ),
]
