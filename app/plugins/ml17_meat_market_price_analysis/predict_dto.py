"""Pydantic DTOs for ml17 (Ridge pork price forecast) /predict endpoint."""
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class PredictBatchRequest(BaseModel):
    """Batch request: CSV with date + 6 exogenous features per row."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["batch"] = "batch"
    data_path: str = Field(
        ..., description="Ruta al CSV con columnas date + 6 features porcino"
    )


class PredictBatchResponse(BaseModel):
    """Batch response: one Ridge prediction per CSV row."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    line: str
    predictions: list[dict[str, Any]]
    output_path: str | None = None


class PredictInlineRequest(BaseModel):
    """Single-row Ridge inference — date + 6 required feature fields."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["inline"] = "inline"
    model_key: str | None = None
    threshold: float | None = None

    date: str = Field(
        ...,
        description="Fecha de referencia (YYYY-MM-DD). Se usa para calcular month_sin/cos.",
    )
    target_price_pigmeat_class_e_es: float = Field(
        ..., description="Precio porcino clase E España en t (€/100 kg) — lag autorregresivo"
    )
    eurostat_pigmeat_slaughter_tonnes_es: float = Field(
        ..., description="Sacrificio porcino España (Eurostat, miles de toneladas)"
    )
    eurostat_pigmeat_slaughter_tonnes_eu: float = Field(
        ..., description="Sacrificio porcino UE (Eurostat, miles de toneladas)"
    )
    cereal_feed_barley_price_monthly: float = Field(
        ..., description="Precio mensual cebada pienso (€/tonelada)"
    )
    cereal_feed_maize_price_monthly: float = Field(
        ..., description="Precio mensual maíz pienso (€/tonelada)"
    )
    mapa_porcino_otras_razas_price_monthly: float = Field(
        ..., description="Precio mensual porcino otras razas MAPA (€/100 kg)"
    )


class PredictInlineResponse(BaseModel):
    """Inline response: predicted pork class E Spain price at t+1."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    line: str
    prediction: Any = Field(..., description="Precio predicho porcino clase E España t+1 (€/100 kg)")
    y_pred: float = Field(..., description="Alias de prediction")
    confidence: float | None = None
    base_date: str = Field(..., description="Fecha de referencia usada (YYYY-MM-DD)")
    xai_feature_values: dict[str, float] | None = Field(
        default=None,
        description="Valores de features usados — consumidos por el servicio XAI",
    )


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictBatchResponse, PredictInlineResponse]
