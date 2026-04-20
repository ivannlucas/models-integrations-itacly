from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, ConfigDict

BUSINESS_TARGETS = ["bovino", "porcino", "ovino", "ave", "carne"]


class MeatPriceRow(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    date: str = Field(..., description="Week date anchored on Monday, format 'YYYY-MM-DD'")
    bovino: float = Field(..., description="IPC index for beef (INE points)")
    porcino: float = Field(..., description="IPC index for pork (INE points)")
    ovino: float = Field(..., description="IPC index for lamb (INE points)")
    ave: float = Field(..., description="IPC index for poultry (INE points)")
    carne: float = Field(..., description="IPC index for meat general (INE points)")


class PredictBatchRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["batch"] = "batch"
    data_path: str = Field(..., description="Path to CSV file with weekly meat price rows")


class PredictBatchResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    predictions: list[dict[str, Any]]
    output_path: str | None = None


class TargetPrediction(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    rf: float = Field(..., description="Random Forest prediction (IPC index points)")
    lstm: float | None = Field(default=None, description="LSTM prediction. None if include_lstm=False.")


class PredictInlineRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["inline"] = "inline"
    model_key: str | None = None
    threshold: float | None = None
    rows: list[MeatPriceRow] = Field(
        ...,
        min_length=4,
        description="Weekly price rows sorted by date. Minimum 4 rows required.",
    )
    include_lstm: bool = Field(default=False, description="If True, also returns LSTM predictions.")


class PredictInlineResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    threshold: float | None = None
    prediction: Any = Field(..., description="Dict of target -> {rf, lstm} predictions")
    confidence: float | None = None
    features_used: list[str]
    prediction_date: str
    rows_used: int
    xai_feature_values: dict[str, float] | None = None


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictBatchResponse, PredictInlineResponse]
