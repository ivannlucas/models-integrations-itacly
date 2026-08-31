"""Pydantic request/response DTOs for the ml16 /predict endpoint."""
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from app.plugins.ml16_meat_raw_material_price_alert.constants import (
    DEFAULT_LOOKBACK,
    FEATURE_WARMUP_ROWS,
)

_MIN_INLINE_ROWS = FEATURE_WARMUP_ROWS + DEFAULT_LOOKBACK + 1  # warmup(6) + lookback(3) + 1 = 10


class PredictBatchRequest(BaseModel):
    """Batch request: a monthly historical CSV, same schema as dataset_clasificacion_base.csv."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["batch"] = "batch"
    data_path: str = Field(
        ...,
        description=(
            "Ruta a un CSV con serie histórica mensual continua: fecha, month, indice_animales, "
            "indice_insumos, precip_total, precip_max, wet_days, wash_days, animales_afectados. "
            f"Se necesitan al menos {_MIN_INLINE_ROWS} meses consecutivos para obtener la primera "
            "predicción (warmup de features + lookback)."
        ),
    )
    mlflow_run_id: str = Field(default="", description="MLflow run ID de un modelo reentrenado por el usuario")


class PredictBatchResponse(BaseModel):
    """Batch response: one row per valid target month."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    predictions: list[dict[str, Any]] = Field(
        ...,
        description=(
            "Una fila por mes objetivo válido: fecha (mes objetivo = último mes de entrada de la "
            "ventana + 4), target_animales_pred/proba/proba_low/proba_high, "
            "target_insumos_pred/proba/proba_low/proba_high."
        ),
    )
    n_predictions: int
    output_path: str | None = None


class PredictInlineRequest(BaseModel):
    """Inline request: the monthly historical series as a list of row dicts."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["inline"] = "inline"
    model_key: str | None = None
    threshold: float | None = Field(
        default=None,
        description=(
            "No usado por este modelo (los umbrales de decisión por target vienen calibrados en "
            "train_config.json, con override manual para target_insumos); se acepta por "
            "compatibilidad de contrato."
        ),
    )
    rows: list[dict[str, Any]] = Field(
        ...,
        min_length=_MIN_INLINE_ROWS,
        description=(
            "Serie histórica mensual (se reordena por 'fecha' internamente): mínimo "
            f"{_MIN_INLINE_ROWS} meses (6 de warmup de features + lookback=3 + 1). Se devuelve la "
            "predicción del mes objetivo más reciente que el historial permite calcular."
        ),
    )
    mlflow_run_id: str = Field(default="", description="MLflow run ID de un modelo reentrenado por el usuario")


class PredictInlineResponse(BaseModel):
    """Inline response: prediction for the most recent target month available."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    fecha: str = Field(
        ..., description="Mes objetivo de la predicción (YYYY-MM-01) = último mes de entrada + 4."
    )
    target_animales_pred: int = Field(..., description="1 = alerta de encarecimiento (XGBoost)")
    target_animales_proba: float
    target_animales_proba_low: float | None = None
    target_animales_proba_high: float | None = None
    target_insumos_pred: int = Field(..., description="1 = alerta de encarecimiento (LogisticRegression)")
    target_insumos_proba: float
    target_insumos_proba_low: float | None = None
    target_insumos_proba_high: float | None = None
    n_rows_used: int
    n_predictions_available: int = Field(
        ..., description="Nº de meses objetivo distintos que el historial recibido permite predecir."
    )
    model_name: str
    xai_feature_values: dict[str, Any] | None = Field(
        default=None, description="Valores usados — consumidos por el servicio de explicabilidad",
    )


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictBatchResponse, PredictInlineResponse]
