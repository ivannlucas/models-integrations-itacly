"""Model registry.

To add a new model:
1. Create app/plugins/<name>/ implementing ModelPluginPort.
2. Add a ModelEntry below — the router and DI are wired automatically.
"""
from dataclasses import dataclass, field
from typing import Any

from app.domain.services.exceptions import (
    InsufficientFramesError,
    InsufficientTelemetryHistoryError,
    InvalidImageError,
    InvalidVideoError,
    NoValidSimulationPointError,
    PuConstraintViolationError,
)

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

from app.plugins.ml2_fungal_cnn_disease_detection.plugin import Ml2FungalCnnDiseaseDetectionPlugin
from app.plugins.ml2_fungal_cnn_disease_detection.predict_dto import (
    PredictRequest as Ml2Fungal_Request,
    PredictResponse as Ml2Fungal_Response,
)

from app.plugins.ml5_meat_cow_behaviour.plugin import Ml5MeatCowBehaviourPlugin
from app.plugins.ml5_meat_cow_behaviour.predict_dto import (
    PredictRequest as Ml5Cow_Request,
    PredictResponse as Ml5Cow_Response,
)

from app.plugins.ml7_cereals_grain_pest_detection.plugin import Ml7CerealsGrainPestDetectionPlugin
from app.plugins.ml7_cereals_grain_pest_detection.predict_dto import (
    PredictRequest as Ml7Grain_Request,
    PredictResponse as Ml7Grain_Response,
)

from app.plugins.ml30_meat_traceability_detection.plugin import Ml30MeatTraceabilityDetectionPlugin
from app.plugins.ml30_meat_traceability_detection.predict_dto import (
    PredictRequest as Ml30Trace_Request,
    PredictResponse as Ml30Trace_Response,
)
from app.plugins.ml30_meat_traceability_detection.train_dto import (
    TrainRequest as Ml30Trace_TrainReq,
    TrainResponse as Ml30Trace_TrainResp,
)

from app.plugins.ml31_cereals_residue_optimizer.plugin import Ml31CerealsResidueOptimizerPlugin
from app.plugins.ml31_cereals_residue_optimizer.predict_dto import (
    PredictRequest as Ml31Residue_Request,
    PredictResponse as Ml31Residue_Response,
)
from app.plugins.ml31_cereals_residue_optimizer.train_dto import (
    TrainRequest as Ml31Residue_TrainReq,
    TrainResponse as Ml31Residue_TrainResp,
)

from app.plugins.ml4_lactic_cnn_thermal_early_disease_detection.plugin import (
    Ml4LacticCnnThermalEarlyDiseaseDetectionPlugin,
)
from app.plugins.ml4_lactic_cnn_thermal_early_disease_detection.predict_dto import (
    PredictRequest as Ml4Thermal_Request,
    PredictResponse as Ml4Thermal_Response,
)

from app.plugins.ml23_lactic_market_price_forecast.plugin import (
    Ml23LacticMarketPriceForecastPlugin,
)
from app.plugins.ml23_lactic_market_price_forecast.predict_dto import (
    PredictRequest as Ml23_Request,
    PredictResponse as Ml23_Response,
)

from app.plugins.ml17_meat_market_price_analysis.plugin import (
    Ml17MeatMarketPriceAnalysisPlugin,
)
from app.plugins.ml17_meat_market_price_analysis.predict_dto import (
    PredictRequest as Ml17_Request,
    PredictResponse as Ml17_Response,
)

from app.plugins.ml35_dairy_ann_cleaning_cost.plugin import Ml35DairyAnnCleaningCostPlugin
from app.plugins.ml35_dairy_ann_cleaning_cost.predict_dto import (
    PredictRequest as Ml35Dairy_Request,
    PredictResponse as Ml35Dairy_Response,
)
from app.plugins.ml35_dairy_ann_cleaning_cost.train_dto import (
    TrainRequest as Ml35Dairy_TrainReq,
    TrainResponse as Ml35Dairy_TrainResp,
)

from app.plugins.ml46_dairy_fouling_clog_detection.plugin import Ml46DairyFoulingClogDetectionPlugin
from app.plugins.ml46_dairy_fouling_clog_detection.predict_dto import (
    PredictRequest as Ml46Dairy_Request,
    PredictResponse as Ml46Dairy_Response,
)
from app.plugins.ml46_dairy_fouling_clog_detection.train_dto import (
    TrainRequest as Ml46Dairy_TrainReq,
    TrainResponse as Ml46Dairy_TrainResp,
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
    ModelEntry(
        model_id="ml5-meat-cow-behaviour",
        prefix="/models/ml5-meat-cow-behaviour",
        version="1.0.0",
        plugin_class=Ml5MeatCowBehaviourPlugin,
        predict_request_type=Ml5Cow_Request,
        predict_response_type=Ml5Cow_Response,
        extra_predict_exceptions=(InvalidVideoError, InvalidImageError, InsufficientFramesError),
    ),
    ModelEntry(
        model_id="ml2-fungal-cnn-disease-detection",
        prefix="/models/ml2-fungal-cnn-disease-detection",
        version="1.0.0",
        plugin_class=Ml2FungalCnnDiseaseDetectionPlugin,
        predict_request_type=Ml2Fungal_Request,
        predict_response_type=Ml2Fungal_Response,
        extra_predict_exceptions=(InvalidImageError,),
    ),
    ModelEntry(
        model_id="ml7-cereals-grain-pest-detection",
        prefix="/models/ml7-cereals-grain-pest-detection",
        version="1.0.0",
        plugin_class=Ml7CerealsGrainPestDetectionPlugin,
        predict_request_type=Ml7Grain_Request,
        predict_response_type=Ml7Grain_Response,
        extra_predict_exceptions=(InvalidImageError,),
    ),
    ModelEntry(
        model_id="ml30-meat-traceability-detection",
        prefix="/models/ml30-meat-traceability-detection",
        version="1.0.0",
        plugin_class=Ml30MeatTraceabilityDetectionPlugin,
        predict_request_type=Ml30Trace_Request,
        predict_response_type=Ml30Trace_Response,
        train_request_type=Ml30Trace_TrainReq,
        train_response_type=Ml30Trace_TrainResp,
    ),
    ModelEntry(
        model_id="ml31-cereals-residue-optimizer",
        prefix="/models/ml31-cereals-residue-optimizer",
        version="1.0.0",
        plugin_class=Ml31CerealsResidueOptimizerPlugin,
        predict_request_type=Ml31Residue_Request,
        predict_response_type=Ml31Residue_Response,
        train_request_type=Ml31Residue_TrainReq,
        train_response_type=Ml31Residue_TrainResp,
    ),
    ModelEntry(
        model_id="ml4-lactic-cnn-thermal-early-disease-detection",
        prefix="/models/ml4-lactic-cnn-thermal-early-disease-detection",
        version="1.0.0",
        plugin_class=Ml4LacticCnnThermalEarlyDiseaseDetectionPlugin,
        predict_request_type=Ml4Thermal_Request,
        predict_response_type=Ml4Thermal_Response,
        extra_predict_exceptions=(InvalidImageError,),
    ),
    ModelEntry(
        model_id="ml23-lactic-market-price-forecast",
        prefix="/models/ml23-lactic-market-price-forecast",
        version="1.0.0",
        plugin_class=Ml23LacticMarketPriceForecastPlugin,
        predict_request_type=Ml23_Request,
        predict_response_type=Ml23_Response,
        extra_predict_exceptions=(),
    ),
    ModelEntry(
        model_id="ml17-meat-market-price-analysis",
        prefix="/models/ml17-meat-market-price-analysis",
        version="1.0.0",
        plugin_class=Ml17MeatMarketPriceAnalysisPlugin,
        predict_request_type=Ml17_Request,
        predict_response_type=Ml17_Response,
        extra_predict_exceptions=(),
    ),
    ModelEntry(
        model_id="ml35-dairy-ann-cleaning-cost",
        prefix="/models/ml35-dairy-ann-cleaning-cost",
        version="1.0.0",
        plugin_class=Ml35DairyAnnCleaningCostPlugin,
        predict_request_type=Ml35Dairy_Request,
        predict_response_type=Ml35Dairy_Response,
        extra_predict_exceptions=(PuConstraintViolationError,),
        train_request_type=Ml35Dairy_TrainReq,
        train_response_type=Ml35Dairy_TrainResp,
    ),
    ModelEntry(
        model_id="ml46-dairy-fouling-clog-detection",
        prefix="/models/ml46-dairy-fouling-clog-detection",
        version="1.0.0",
        plugin_class=Ml46DairyFoulingClogDetectionPlugin,
        predict_request_type=Ml46Dairy_Request,
        predict_response_type=Ml46Dairy_Response,
        extra_predict_exceptions=(InsufficientTelemetryHistoryError,),
        train_request_type=Ml46Dairy_TrainReq,
        train_response_type=Ml46Dairy_TrainResp,
    ),
]
