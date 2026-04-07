"""Model registry.

To add a new model:
1. Create model-runtime-<name>/ with the standard structure (standalone deployment).
2. Create app/plugins/<name>/ with the plugin code (imports adjusted for the central app).
3. Add a ModelEntry below — that's all.
"""
from dataclasses import dataclass, field
from typing import Any

from app.domain.services.exceptions import (
    InsufficientDataError,
    InsufficientFramesError,
    InsufficientRowsError,
    InvalidImageError,
    InvalidVideoError,
    NoValidSimulationPointError,
    UnsupportedProductError,
)

# ── Plugin imports ────────────────────────────────────────────────────────────

from app.plugins.wine_price_fluctuation.plugin import WinePriceFluctuationPlugin
from app.plugins.wine_price_fluctuation.predict_dto import (
    PredictBatchResponse as WinePF_BatchResp,
    PredictInlineResponse as WinePF_InlineResp,
    PredictRequest as WinePF_Request,
    PredictResponse as WinePF_Response,
)

from app.plugins.cereal_price_forecast.plugin import CerealPriceForecastPlugin
from app.plugins.cereal_price_forecast.predict_dto import (
    PredictBatchResponse as Cereal_BatchResp,
    PredictInlineResponse as Cereal_InlineResp,
    PredictRequest as Cereal_Request,
    PredictResponse as Cereal_Response,
)

from app.plugins.meat_price_forecast.plugin import MeatPriceForecastPlugin
from app.plugins.meat_price_forecast.predict_dto import (
    PredictBatchResponse as Meat_BatchResp,
    PredictInlineResponse as Meat_InlineResp,
    PredictRequest as Meat_Request,
    PredictResponse as Meat_Response,
)

from app.plugins.cnn_fungal_detection.plugin import CnnFungalDetectionPlugin
from app.plugins.cnn_fungal_detection.predict_dto import (
    PredictBatchResponse as CnnFungal_BatchResp,
    PredictInlineResponse as CnnFungal_InlineResp,
    PredictRequest as CnnFungal_Request,
    PredictResponse as CnnFungal_Response,
)

from app.plugins.cnn_thermal_scm.plugin import CnnThermalScmPlugin
from app.plugins.cnn_thermal_scm.predict_dto import (
    PredictBatchResponse as CnnThermal_BatchResp,
    PredictInlineResponse as CnnThermal_InlineResp,
    PredictRequest as CnnThermal_Request,
    PredictApiResponse as CnnThermal_Response,
)

from app.plugins.cow_behavior.plugin import CowBehaviorPlugin
from app.plugins.cow_behavior.predict_dto import (
    PredictBatchResponse as Cow_BatchResp,
    PredictInlineResponse as Cow_InlineResp,
    PredictRequest as Cow_Request,
    PredictResponse as Cow_Response,
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


# ── Registry ──────────────────────────────────────────────────────────────────

REGISTRY: list[ModelEntry] = [
    ModelEntry(
        model_id="wine-price-fluctuation",
        prefix="/models/wine-price-fluctuation",
        version="1.0.0",
        plugin_class=WinePriceFluctuationPlugin,
        predict_request_type=WinePF_Request,
        predict_response_type=WinePF_Response,
        batch_response_class=WinePF_BatchResp,
        inline_response_class=WinePF_InlineResp,
        extra_predict_exceptions=(InsufficientDataError,),
    ),
    ModelEntry(
        model_id="cereal-price-forecast",
        prefix="/models/cereal-price-forecast",
        version="1.0.0",
        plugin_class=CerealPriceForecastPlugin,
        predict_request_type=Cereal_Request,
        predict_response_type=Cereal_Response,
        batch_response_class=Cereal_BatchResp,
        inline_response_class=Cereal_InlineResp,
        extra_predict_exceptions=(UnsupportedProductError,),
    ),
    ModelEntry(
        model_id="meat-price-forecast",
        prefix="/models/meat-price-forecast",
        version="1.0.0",
        plugin_class=MeatPriceForecastPlugin,
        predict_request_type=Meat_Request,
        predict_response_type=Meat_Response,
        batch_response_class=Meat_BatchResp,
        inline_response_class=Meat_InlineResp,
        extra_predict_exceptions=(InsufficientRowsError,),
    ),
    ModelEntry(
        model_id="cnn-fungal-detection",
        prefix="/models/cnn-fungal-detection",
        version="1.0.0",
        plugin_class=CnnFungalDetectionPlugin,
        predict_request_type=CnnFungal_Request,
        predict_response_type=CnnFungal_Response,
        batch_response_class=CnnFungal_BatchResp,
        inline_response_class=CnnFungal_InlineResp,
        extra_predict_exceptions=(InvalidImageError,),
    ),
    ModelEntry(
        model_id="cnn-thermal-scm",
        prefix="/models/cnn-thermal-scm",
        version="1.0.0",
        plugin_class=CnnThermalScmPlugin,
        predict_request_type=CnnThermal_Request,
        predict_response_type=CnnThermal_Response,
        batch_response_class=CnnThermal_BatchResp,
        inline_response_class=CnnThermal_InlineResp,
        extra_predict_exceptions=(InvalidImageError,),
    ),
    ModelEntry(
        model_id="cow-behavior",
        prefix="/models/cow-behavior",
        version="1.0.0",
        plugin_class=CowBehaviorPlugin,
        predict_request_type=Cow_Request,
        predict_response_type=Cow_Response,
        batch_response_class=Cow_BatchResp,
        inline_response_class=Cow_InlineResp,
        extra_predict_exceptions=(InvalidVideoError, InvalidImageError, InsufficientFramesError),
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
