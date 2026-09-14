"""Pydantic request/response DTOs for the ml15 /predict endpoint.

PredictInlineRequest declares each of the artifact's 16 feature_columns as its own top-level
field (Swagger-documented, typed) — PredictModelUseCase.execute() builds the plugin's
``features`` dict via ``request.model_dump(exclude={"mode", "model_key", "threshold",
"mlflow_run_id", "data_path"})``, so a generic nested ``features: dict`` field here would end up
double-wrapped (``{"features": {...}}``) instead of the flat dict plugin.py expects — this is
the same convention already used by every other plugin in the repo (see e.g.
ml17_meat_market_price_analysis/predict_dto.py).
"""
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

_CALENDAR_NOTE = "Opcional: se deriva de 'date'/'origin_date' si se omite."


class PredictBatchRequest(BaseModel):
    """Batch request: a CSV with one row per prediction, same feature contract as inline."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["batch"] = "batch"
    data_path: str = Field(
        ...,
        description=(
            "Ruta a un CSV con una fila por predicción: ipi_national_current, "
            "ipi_national_lag_1..6, chem_sector_lag_11, copper_lag_14, eur_usd_lag_17, "
            "oil_brent_lag_12, usa_lag_1 (índice base 2020=100), más 'date' (YYYY-MM-DD) o las 4 "
            "columnas de calendario ya calculadas (month_sin, month_cos, quarter, is_spring_risk)."
        ),
    )
    mlflow_run_id: str = Field(default="", description="MLflow run ID de un modelo reentrenado por el usuario")


class PredictBatchResponse(BaseModel):
    """Batch response: one row per input row (y_pred, or 'error' if that row failed)."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    predictions: list[dict[str, Any]] = Field(
        ...,
        description="Una entrada por fila: row, origin_date/target_date (si derivables), y_pred, y_anchor.",
    )
    n_predictions: int
    output_path: str | None = None


class PredictInlineRequest(BaseModel):
    """Inline request: the artifact's 16 feature_columns, one field each."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["inline"] = "inline"
    model_key: str | None = None
    threshold: float | None = Field(
        default=None, description="No usado por este modelo; se acepta por compatibilidad de contrato."
    )

    date: str | None = Field(
        default=None,
        description="Fecha de origen (YYYY-MM-DD). Alias de 'origin_date'. " + _CALENDAR_NOTE,
    )
    origin_date: str | None = Field(default=None, description="Alias de 'date'.")

    ipi_national_current: float = Field(
        ..., description="Último IPI fitosanitario nacional observado en origin_date (índice base 2020=100)."
    )
    ipi_national_lag_1: float = Field(..., description="IPI nacional observado 1 mes antes de origin_date.")
    ipi_national_lag_2: float = Field(..., description="IPI nacional observado 2 meses antes de origin_date.")
    ipi_national_lag_3: float = Field(..., description="IPI nacional observado 3 meses antes de origin_date.")
    ipi_national_lag_4: float = Field(..., description="IPI nacional observado 4 meses antes de origin_date.")
    ipi_national_lag_5: float = Field(..., description="IPI nacional observado 5 meses antes de origin_date.")
    ipi_national_lag_6: float = Field(..., description="IPI nacional observado 6 meses antes de origin_date.")
    chem_sector_lag_11: float = Field(..., description="Índice sectorial químico retardado 11 meses (base 2020=100).")
    copper_lag_14: float = Field(..., description="Precio del cobre retardado 14 meses (base 2020=100).")
    eur_usd_lag_17: float = Field(..., description="Tipo de cambio EUR/USD retardado 17 meses (base 2020=100).")
    oil_brent_lag_12: float = Field(..., description="Precio del petróleo Brent retardado 12 meses (base 2020=100).")
    usa_lag_1: float = Field(..., description="Índice de precios de pesticidas de EE.UU. retardado 1 mes (base 2020=100).")

    month_sin: float | None = Field(default=None, description="sin(2*pi*mes/12). " + _CALENDAR_NOTE)
    month_cos: float | None = Field(default=None, description="cos(2*pi*mes/12). " + _CALENDAR_NOTE)
    quarter: int | None = Field(default=None, description="Trimestre natural (1-4). " + _CALENDAR_NOTE)
    is_spring_risk: int | None = Field(
        default=None,
        description="1 si abril/mayo/junio (riesgo fúngico primaveral vid), 0 en otro caso. " + _CALENDAR_NOTE,
    )

    mlflow_run_id: str = Field(default="", description="MLflow run ID de un modelo reentrenado por el usuario")


class PredictInlineResponse(BaseModel):
    """Inline response: t+6 national IPI forecast for the given feature vector."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    origin_date: str | None = Field(
        default=None, description="Fecha de origen (YYYY-MM-01), si se pudo derivar de 'date'/'origin_date'."
    )
    target_date: str | None = Field(
        default=None, description="origin_date + horizon (YYYY-MM-01), si origin_date es derivable."
    )
    horizon: int = Field(default=6, description="Horizonte productivo vigente (siempre 6).")
    y_pred: float = Field(..., description="IPI fitosanitario nacional predicho a t+6 (índice base 2020=100).")
    y_anchor: float = Field(..., description="ipi_national_current usado como ancla (referencia de dirección).")
    model_name: str
    xai_feature_values: dict[str, float] | None = Field(
        default=None, description="Valores de las 16 features usadas — consumido por el servicio de explicabilidad."
    )


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictBatchResponse, PredictInlineResponse]
