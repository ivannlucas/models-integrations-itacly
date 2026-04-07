from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field


class WeeklyPriceRecord(BaseModel):
    campaign: str = Field(
        ...,
        description="Viticulture campaign in 'YYYY/YYYY+1' format, e.g. '2023/2024'",
    )
    week: int = Field(..., ge=1, le=52, description="ISO week number (1-52)")
    price_red: float = Field(..., gt=0, description="Red wine price in EUR/hl, e.g. 42.50")


class PredictBatchRequest(BaseModel):
    mode: Literal["batch"] = "batch"
    data_path: str = Field(
        ..., description="Path to CSV file with weekly price records inside the container"
    )


class PredictBatchResponse(BaseModel):
    model_id: str
    predictions: list[dict[str, Any]]
    output_path: str | None = None


class PredictInlineRequest(BaseModel):
    mode: Literal["inline"] = "inline"
    model_key: str | None = None
    threshold: float | None = None
    records: list[WeeklyPriceRecord] = Field(
        ...,
        min_length=22,
        description=(
            "List of weekly price records. Minimum 22 records required to compute "
            "all technical indicators (Bollinger band requires 20-week lookback)."
        ),
    )


class PredictInlineResponse(BaseModel):
    model_id: str
    threshold: float | None = None
    prediction: Any = Field(
        ..., description="Binary class: 1 = price rises >= 2.5% in next 4 weeks, 0 = otherwise"
    )
    confidence: float | None = Field(
        None, description="Probability of class 1 (range 0.0 to 1.0)"
    )
    features_used: list[str]
    model_type: str = Field(..., description="Model type used: 'logreg' or 'xgboost'")
    prediction_date: str = Field(
        ..., description="ISO date string of the week the prediction refers to"
    )
    xai_feature_values: dict[str, float] | None = Field(
        default=None,
        description="Actual feature values used for inference — consumed by the XAI service",
    )


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictBatchResponse, PredictInlineResponse]
