"""Pydantic request/response DTOs for the ml18 /predict endpoint."""
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from app.plugins.ml18_meat_spatial_price_forecast.constants import LOOKBACK


class PredictBatchRequest(BaseModel):
    """Batch request: a CSV with a panel of monthly rows (one or more CCAA-Producto groups)."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["batch"] = "batch"
    data_path: str = Field(
        ...,
        description=(
            "Ruta a un CSV (separador ';') con panel mensual: Fecha, CCAA, Producto, Poblacion, "
            "RentaHogar, CONSUMO X CAPITA, PENETRACION (%), PRECIO MEDIO KG. Mínimo "
            f"{LOOKBACK} meses consecutivos por combinación CCAA-Producto a predecir."
        ),
    )


class PredictBatchResponse(BaseModel):
    """Batch response: one next-month prediction per (CCAA, Producto) group with enough history."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    predictions: list[dict[str, Any]]
    n_predictions: int
    output_path: str | None = None


class PredictInlineRequest(BaseModel):
    """Inline request: the panel as a list of row dicts."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["inline"] = "inline"
    model_key: str | None = None
    threshold: float | None = Field(
        default=None,
        description="No usado por este modelo (no hay umbral de decisión); se acepta por compatibilidad de contrato.",
    )
    rows: list[dict[str, Any]] = Field(
        ...,
        min_length=LOOKBACK,
        description=(
            "Panel mensual: Fecha, CCAA, Producto, Poblacion, RentaHogar, CONSUMO X CAPITA, "
            "PENETRACION (%), PRECIO MEDIO KG (opcional solo en la última fila conocida). "
            f"Cada combinación CCAA-Producto que se quiera predecir necesita al menos {LOOKBACK} "
            "meses consecutivos propios; el histórico de CCAA vecinas (mismo Producto, mismas "
            "fechas) es opcional pero mejora la señal espacial del modelo."
        ),
    )


class PredictInlineResponse(BaseModel):
    """Inline response: one next-month prediction per (CCAA, Producto) group with enough history."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    predictions: list[dict[str, Any]] = Field(
        ...,
        description="Lista de {CCAA, Producto, Fecha (mes predicho), predicted_price (€/kg)}.",
    )
    n_predictions: int


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictBatchResponse, PredictInlineResponse]
