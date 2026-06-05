"""Model registry.

To add a new model:
1. Create app/plugins/<name>/ implementing ModelPluginPort.
2. Add a ModelEntry below — the router and DI are wired automatically.
"""
from dataclasses import dataclass, field
from typing import Any

from app.domain.services.exceptions import NoValidSimulationPointError
from app.application.dto.train_dto import TrainResponse, TrainRequest

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

from app.plugins.modelo10_lacteo.plugin import Modelo10LacteoPlugin
from app.plugins.modelo10_lacteo.predict_dto import (
    PredictBatchResponse as Lacteo10_BatchResp,
    PredictInlineResponse as Lacteo10_InlineResp,
    PredictRequest as Lacteo10_Request,
    PredictResponse as Lacteo10_Response,
)


# ── Registry entry dataclass ──────────────────────────────────────────────────

@dataclass
class ModelEntry:
    """Defines the metadata and types for a model plugin."""
    model_id: str
    prefix: str
    version: str
    plugin_class: type
    predict_request_type: Any
    predict_response_type: Any
    batch_response_class: type
    inline_response_class: type
    # Optional values, you need to leave them at the bottom
    train_request_type: Any | None = TrainRequest
    train_response_type: Any | None = TrainResponse
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
    ModelEntry(
        model_id="modelo10-lacteo",
        prefix="/models/modelo10-lacteo",
        version="1.0.0",
        plugin_class=Modelo10LacteoPlugin,
        predict_request_type=Lacteo10_Request,
        predict_response_type=Lacteo10_Response,
        batch_response_class=Lacteo10_BatchResp,
        inline_response_class=Lacteo10_InlineResp,
        extra_predict_exceptions=(),
    ),
]
