"""Pydantic DTOs for m21 (ESP-CEREAL spatial cereal price) /predict endpoint."""
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class PredictBatchRequest(BaseModel):
    """Batch request: CSV with raw panel rows for a given month."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["batch"] = "batch"
    data_path: str = Field(
        ..., description="Ruta al CSV con filas del panel cerealístico"
    )
    month: str = Field(
        ..., description="Mes de predicción (YYYY-MM)"
    )
    role: str = Field(
        default="comprador",
        description="Rol del usuario: 'comprador' o 'vendedor'",
    )


class PredictBatchResponse(BaseModel):
    """Batch response: predictions for all province×cereal combinations in the month."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    predictions: list[dict[str, Any]]
    output_path: str | None = None


class PredictInlineRequest(BaseModel):
    """Single-row inference: province + cereal + month → 3-horizon predictions."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["inline"] = "inline"
    model_key: str | None = None
    threshold: float | None = None

    provincia: str = Field(..., description="Provincia española (e.g., 'Burgos')")
    cereal_predominante: str = Field(..., description="Tipo de cereal ('trigo', 'cebada', 'maíz')")
    date: str = Field(..., description="Mes de predicción (YYYY-MM)")
    role: str = Field(
        default="comprador",
        description="Rol del usuario: 'comprador' o 'vendedor'",
    )

    # Optional raw features — if omitted, the plugin tries to load from the dataset
    lat_centroide: float | None = None
    lon_centroide: float | None = None
    dist_puerto_min_km: float | None = None
    presion_combustible_puerto: float | None = None


class PredictInlineResponse(BaseModel):
    """Inline response: 3-horizon predictions with signals, confidence, and recommendation."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    province: str
    cereal: str
    month: str
    geo_risk: bool
    timing_label: str
    causal_drivers: list[str]
    card_text: str
    predictions: dict[str, dict[str, Any]]
    model_version: str = Field(default="1.0.0")
    xai_feature_values: dict[str, float] | None = Field(
        default=None,
        description="Valores de features usados — consumidos por el servicio XAI",
    )


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictBatchResponse, PredictInlineResponse]
