from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, ConfigDict

VALID_PRODUCTS = frozenset(
    ["Durum wheat", "Milling wheat", "Feed barley", "Malting barley", "Feed maize"]
)


class PredictBatchRequest(BaseModel):
    mode: Literal["batch"] = "batch"
    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(
        ...,
        description="Path to CSV file with pre-computed cereal price features inside the container",
    )


class PredictBatchResponse(BaseModel):
    model_id: str
    model_config = ConfigDict(protected_namespaces=())
    predictions: list[dict[str, Any]]
    output_path: str | None = None


class PredictInlineRequest(BaseModel):
    mode: Literal["inline"] = "inline"
    model_config = ConfigDict(protected_namespaces=())
    model_key: str | None = None
    threshold: float | None = None
    product_name: str = Field(
        ...,
        description=(
            "Cereal product name. Valid values: "
            "'Durum wheat', 'Milling wheat', 'Feed barley', 'Malting barley', 'Feed maize'"
        ),
    )
    market_name: str | None = Field(default="Unknown", description="Spanish market name")
    week_begin_date: str | None = Field(default="Unknown", description="ISO 8601 week start date")

    Year: float | None = None
    Month: float | None = None
    Quarter: float | None = None
    Week_of_Year: float | None = None
    week_index: int | None = None
    month_sin: float | None = None
    month_cos: float | None = None
    week_sin: float | None = None
    week_cos: float | None = None
    prec: float | None = None
    tmed: float | None = None
    tmin: float | None = None
    tmax: float | None = None
    prec_lag_1w: float | None = None
    prec_lag_2w: float | None = None
    prec_lag_4w: float | None = None
    prec_lag_8w: float | None = None
    tmin_lag_1w: float | None = None
    tmin_lag_2w: float | None = None
    tmin_lag_4w: float | None = None
    tmin_lag_8w: float | None = None
    tmed_lag_1w: float | None = None
    tmed_lag_2w: float | None = None
    tmed_lag_4w: float | None = None
    tmed_lag_8w: float | None = None
    tmax_lag_1w: float | None = None
    tmax_lag_2w: float | None = None
    tmax_lag_4w: float | None = None
    tmax_lag_8w: float | None = None
    prec_rolling_mean_4w: float | None = None
    prec_rolling_mean_12w: float | None = None
    tmin_rolling_mean_4w: float | None = None
    tmin_rolling_mean_12w: float | None = None
    tmed_rolling_mean_4w: float | None = None
    tmed_rolling_mean_12w: float | None = None
    tmax_rolling_mean_4w: float | None = None
    tmax_rolling_mean_12w: float | None = None
    prec_rolling_sum_4w: float | None = None
    prec_rolling_sum_12w: float | None = None
    prec_anomaly: float | None = None
    tmed_anomaly: float | None = None
    Fertilizers_index: float | None = None
    Seeds_index: float | None = None
    fertilizers_lag_0w: float | None = None
    fertilizers_lag_4w: float | None = None
    fertilizers_lag_8w: float | None = None
    fertilizers_lag_12w: float | None = None
    seeds_lag_0w: float | None = None
    seeds_lag_4w: float | None = None
    seeds_lag_8w: float | None = None
    seeds_lag_12w: float | None = None
    fertilizers_rolling_mean_12w: float | None = None
    fertilizers_rolling_mean_26w: float | None = None
    seeds_rolling_mean_12w: float | None = None
    seeds_rolling_mean_26w: float | None = None
    fertilizers_x_product_Durum_wheat: float | None = None
    fertilizers_x_product_Feed_barley: float | None = None
    fertilizers_x_product_Feed_maize: float | None = None
    fertilizers_x_product_Feed_wheat: float | None = None
    fertilizers_x_product_Malting_barley: float | None = None
    fertilizers_x_product_Milling_wheat: float | None = None
    seeds_x_product_Durum_wheat: float | None = None
    seeds_x_product_Feed_barley: float | None = None
    seeds_x_product_Feed_maize: float | None = None
    seeds_x_product_Feed_wheat: float | None = None
    seeds_x_product_Malting_barley: float | None = None
    seeds_x_product_Milling_wheat: float | None = None
    price_lag_1w: float | None = None
    price_lag_2w: float | None = None
    price_lag_4w: float | None = None
    price_lag_8w: float | None = None
    price_rolling_mean_4w: float | None = None
    price_rolling_mean_12w: float | None = None
    price_rolling_std_4w: float | None = None


class PredictInlineResponse(BaseModel):
    model_id: str
    model_config = ConfigDict(protected_namespaces=())
    threshold: float | None = None
    prediction: Any = Field(..., description="Predicted market price (EUR/tonne)")
    confidence: float | None = None
    features_used: list[str]
    product_name: str
    market_name: str
    week_begin_date: str
    model_version: str = Field(default="1.0")
    xai_feature_values: dict[str, float] | None = None


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictBatchResponse, PredictInlineResponse]
