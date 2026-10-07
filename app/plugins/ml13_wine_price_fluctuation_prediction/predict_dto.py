"""Pydantic request/response DTOs for the ml13 /predict endpoint."""
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.plugins.ml13_wine_price_fluctuation_prediction.constants import (
    BULLETIN_COLUMN,
    CAMPAIGN_COLUMN,
    MIN_INFERENCE_WEEKS,
    PRICE_COLUMN_CANDIDATES,
    WEEK_COLUMN,
)

_SCHEMA_DOC = (
    "Serie semanal de precios del vino a granel (boletín MAPA) con 'campaign' (p. ej. '2024/2025') "
    "+ 'week' (semana ISO) o, alternativamente, 'bulletin' ('SEMANA {semana}/{año_fin}', donde "
    "año_fin es el SEGUNDO año de la campaña: SEMANA 46/2026 = semana 46 de 2025), y una columna "
    f"de precio en EUR/hl ({', '.join(PRICE_COLUMN_CANDIDATES)}; se usa la primera presente). "
    f"Se necesitan al menos {MIN_INFERENCE_WEEKS} semanas válidas (ventana de Bollinger-20)."
)


class PredictBatchRequest(BaseModel):
    """Batch request: a raw weekly price CSV, same schema as data/raw/mapa_wine_prices_raw.csv."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["batch"] = "batch"
    data_path: str = Field(..., description=f"Ruta local o s3:// a un CSV. {_SCHEMA_DOC}")
    mlflow_run_id: str = Field(
        default="", description="MLflow run ID de un modelo reentrenado por el usuario",
    )


class PredictBatchResponse(BaseModel):
    """Batch response: one row per input row (predict_from_csv keeps every input row)."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    model_type: str = Field(
        ..., description="logreg | xgboost (según model_config.json del artefacto usado)",
    )
    predictions: list[dict[str, Any]] = Field(
        ...,
        description=(
            "Una fila por fila de entrada: fecha (lunes de la semana ISO de referencia t), "
            "bulletin/campaign/week, price, las 6 features y pred_proba_up = P(media del precio en "
            "t+1..t+4 > precio_t * 1.025). Las filas del warm-up inicial (~19 semanas) o sin "
            "precio llevan pred_proba_up = null."
        ),
    )
    n_rows: int
    n_predictions: int = Field(..., description="Filas con pred_proba_up calculada")
    decision_threshold: float
    output_path: str | None = None


class PredictInlineRequest(BaseModel):
    """Inline request: the weekly price history as a list of row dicts."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["inline"] = "inline"
    model_key: str | None = None
    threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Umbral de decisión para 'alerta_subida' (por defecto 0.5, el del código original). "
            "No altera pred_proba_up."
        ),
    )
    rows: list[dict[str, Any]] = Field(
        ...,
        min_length=MIN_INFERENCE_WEEKS,
        description=f"{_SCHEMA_DOC} Se devuelve la predicción de la semana más reciente.",
    )
    mlflow_run_id: str = Field(
        default="", description="MLflow run ID de un modelo reentrenado por el usuario",
    )

    @field_validator("rows")
    @classmethod
    def _check_schema(cls, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        keys = set(rows[0]) if rows else set()
        if not ({CAMPAIGN_COLUMN, WEEK_COLUMN} <= keys or BULLETIN_COLUMN in keys):
            raise ValueError(
                f"cada fila necesita '{CAMPAIGN_COLUMN}'+'{WEEK_COLUMN}' o '{BULLETIN_COLUMN}'"
            )
        if not keys.intersection(PRICE_COLUMN_CANDIDATES):
            raise ValueError(f"cada fila necesita una columna de precio: {PRICE_COLUMN_CANDIDATES}")
        return rows


class PredictInlineResponse(BaseModel):
    """Inline response: prediction for the most recent week of the submitted history."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    fecha: str = Field(..., description="Lunes de la semana ISO de referencia t (YYYY-MM-DD)")
    campaign: str | None = None
    week: int | None = None
    price: float = Field(..., description="Precio de la semana t (EUR/hl)")
    pred_proba_up: float = Field(
        ..., description="P(media del precio en t+1..t+4 > precio_t * (1 + 0.025))"
    )
    alerta_subida: int = Field(..., description="1 si pred_proba_up >= decision_threshold")
    decision_threshold: float
    horizon_weeks: int
    return_threshold: float
    n_rows_used: int
    n_predictions_available: int = Field(
        ..., description="Semanas del historial con predicción calculable",
    )
    model_name: str
    model_type: str
    xai_feature_values: dict[str, Any] | None = Field(
        default=None,
        description="Valores (sin escalar) de las 6 features usadas — servicio de explicabilidad",
    )


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictBatchResponse, PredictInlineResponse]
