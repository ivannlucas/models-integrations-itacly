"""Model registry.

To add a new model:
1. Create model-runtime-<name>/ with the standard structure (standalone deployment).
2. Create app/plugins/<name>/ with the plugin code (imports adjusted for the central app).
3. Add a ModelEntry below — that's all.
"""
from dataclasses import dataclass, field
from typing import Any

from app.domain.services.exceptions import InvalidImageError, NoValidSimulationPointError

# ── Plugin imports ────────────────────────────────────────────────────────────


from app.plugins.ml8_cereals_img_anomaly_detector.plugin import Ml8CerealsImgAnomalyDetectorPlugin
from app.plugins.ml8_cereals_img_anomaly_detector.predict_dto import (
    PredictBatchResponse as Ml8CerealsImgAnomalyDetector_BatchResp,
    PredictInlineResponse as Ml8CerealsImgAnomalyDetector_InlineResp,
    PredictRequest as Ml8CerealsImgAnomalyDetector_Request,
    PredictResponse as Ml8CerealsImgAnomalyDetector_Response,
)
from app.plugins.ml8_cereals_img_anomaly_detector.train_dto import (
    TrainRequest as Ml8CerealsImgAnomalyDetector_TrainReq,
    TrainResponse as Ml8CerealsImgAnomalyDetector_TrainResp,
)
from app.plugins.wine_sulphite.plugin import WineSulphitePlugin
from app.plugins.wine_sulphite.predict_dto import (
    PredictBatchResponse as WineSO2_BatchResp,
    PredictInlineResponse as WineSO2_InlineResp,
    PredictRequest as WineSO2_Request,
    PredictResponse as WineSO2_Response,
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
    extra_predict_exceptions: tuple[type[Exception], ...] = field(default_factory=tuple)
    train_request_type: Any = None
    train_response_type: Any = None


# ── Registry ──────────────────────────────────────────────────────────────────

REGISTRY: list[ModelEntry] = [
    ModelEntry(
        model_id="ml8-cereals-img-anomaly-detector",
        prefix="/models/ml8-cereals-img-anomaly-detector",
        version="1.0.0",
        plugin_class=Ml8CerealsImgAnomalyDetectorPlugin,
        predict_request_type=Ml8CerealsImgAnomalyDetector_Request,
        predict_response_type=Ml8CerealsImgAnomalyDetector_Response,
        batch_response_class=Ml8CerealsImgAnomalyDetector_BatchResp,
        inline_response_class=Ml8CerealsImgAnomalyDetector_InlineResp,
        extra_predict_exceptions=(InvalidImageError,),
        train_request_type=Ml8CerealsImgAnomalyDetector_TrainReq,
        train_response_type=Ml8CerealsImgAnomalyDetector_TrainResp,
    ),
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
]
