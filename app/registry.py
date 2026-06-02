"""Model registry.

To add a new model:
1. Create model-runtime-<name>/ with the standard structure (standalone deployment).
2. Create app/plugins/<name>/ with the plugin code (imports adjusted for the central app).
3. Add a ModelEntry below — that's all.
"""
from dataclasses import dataclass, field
from typing import Any

from app.domain.services.exceptions import NoValidSimulationPointError
from app.application.dto.train_dto import TrainResponse, TrainRequest

# ── Plugin imports ────────────────────────────────────────────────────────────


from app.plugins.wine_sulphite.plugin import WineSulphitePlugin
from app.plugins.wine_sulphite.predict_dto import (
    PredictBatchResponse as WineSO2_BatchResp,
    PredictInlineResponse as WineSO2_InlineResp,
    PredictRequest as WineSO2_Request,
    PredictResponse as WineSO2_Response,
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


# ── Registry ──────────────────────────────────────────────────────────────────

REGISTRY: list[ModelEntry] = [
    ModelEntry(
        model_id="wine-sulphite",
        prefix="/models/wine-sulphite",
        version="1.2.0",
        plugin_class=WineSulphitePlugin,
        predict_request_type=WineSO2_Request,
        predict_response_type=WineSO2_Response,
        batch_response_class=WineSO2_BatchResp,
        inline_response_class=WineSO2_InlineResp,
        extra_predict_exceptions=(NoValidSimulationPointError,),
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
