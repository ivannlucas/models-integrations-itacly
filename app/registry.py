"""Model registry.

To add a new model:
1. Create app/plugins/<name>/ implementing ModelPluginPort.
2. Add a ModelEntry below — the router and DI are wired automatically.
"""
from dataclasses import dataclass, field
from typing import Any

from app.domain.services.exceptions import InvalidImageError, NoValidSimulationPointError

# ── Plugin imports ────────────────────────────────────────────────────────────


from app.plugins.ml25_wine_sulphites.plugin import WineSulphitePlugin
from app.plugins.ml25_wine_sulphites.predict_dto import (
    PredictRequest as WineSO2_Request,
    PredictResponse as WineSO2_Response,
)
from app.plugins.ml25_wine_sulphites.train_dto import (
    TrainRequest as WineSO2_TrainReq,
    TrainResponse as WineSO2_TrainResp,
)

from app.plugins.modelo10_lacteo.plugin import Modelo10LacteoPlugin
from app.plugins.modelo10_lacteo.predict_dto import (
    PredictRequest as Lacteo10_Request,
    PredictResponse as Lacteo10_Response,
)

from app.plugins.ml8_cereals_img_anomaly_detector.plugin import Ml8CerealsImgAnomalyDetectorPlugin
from app.plugins.ml8_cereals_img_anomaly_detector.predict_dto import (
    PredictRequest as Ml8Cereals_Request,
    PredictResponse as Ml8Cereals_Response,
)
from app.plugins.ml8_cereals_img_anomaly_detector.train_dto import (
    TrainRequest as Ml8Cereals_TrainReq,
    TrainResponse as Ml8Cereals_TrainResp,
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
    train_request_type: Any | None = None
    train_response_type: Any | None = None
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
        extra_predict_exceptions=(),
    ),
    ModelEntry(
        model_id="ml8-cereals-img-anomaly-detector",
        prefix="/models/ml8-cereals-img-anomaly-detector",
        version="1.0.0",
        plugin_class=Ml8CerealsImgAnomalyDetectorPlugin,
        predict_request_type=Ml8Cereals_Request,
        predict_response_type=Ml8Cereals_Response,
        extra_predict_exceptions=(InvalidImageError,),
        train_request_type=Ml8Cereals_TrainReq,
        train_response_type=Ml8Cereals_TrainResp,
    ),
]
