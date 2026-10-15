"""Pydantic request/response DTOs for the ml24 /predict endpoint."""
# pylint: disable=line-too-long  # repo limit is 120 (.flake8), enforced by flake8
# pylint: disable=too-few-public-methods  # Pydantic DTOs carry no behaviour
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

_CCAA_DESC = (
    "CCAA a devolver: códigos de 3 letras de src/utils/ccaa_codes.py (AND, ARA, NAV, CAN, CTB, CLM, "
    "CYL, CAT, MAD, VAL, EXT, GAL, BAL, RIO, PVA, AST, MUR) o el nombre oficial. Vacío = las 17. "
    "El modelo se calcula siempre con todas (las señales espaciales usan las vecinas)."
)
_HORIZONS_DESC = "Horizontes en meses a devolver (subconjunto de [1, 3, 6]). Vacío = los tres."
_AS_OF_DESC = (
    "Opcional, 'YYYY-MM': usa la serie solo hasta ese mes (equivale a ejecutar el predictor original "
    "con un --input que termina ahí). Sirve para backtesting; por defecto, el último mes disponible."
)


_RUN_DESC = "MLflow run ID de un modelo reentrenado por el usuario (vacío = modelo base)."


class PredictBatchRequest(BaseModel):
    """Batch request: a long-format MAPA series CSV (the original's --input), every CCAA x horizon.

    The generic use case only forwards data_path/mlflow_run_id to predict_batch, so the
    ccaa/horizons/as_of_period filters exist in inline mode only.
    """

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["batch"] = "batch"
    mlflow_run_id: str = Field(default="", description=_RUN_DESC)
    data_path: str = Field(
        ...,
        description=(
            "Ruta local o s3:// a un CSV en el formato de series_consumo_lacteos_ccaa.csv "
            "(scripts/data_processing.py): anio, mes, ccaa, producto, metrica, valor, es_reconstruido. "
            "Debe incluir TOTAL LECHE LIQUIDA con 'PRECIO MEDIO kg o litros' y los 4 productos de "
            "contexto, con al menos 13 meses por CCAA. Las exógenas y la matriz espacial son las del "
            "artefacto base. Devuelve las 17 CCAA x 3 horizontes (51 filas), como el predictor original."
        ),
    )


class PredictInlineRequest(BaseModel):
    """Inline request: predict from the base series bundled with the artifacts."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["inline"] = "inline"
    model_key: str | None = None
    threshold: float | None = Field(
        default=None,
        description="No usado por este modelo (no hay umbral de decisión); se acepta por compatibilidad de contrato.",
    )
    mlflow_run_id: str = Field(default="", description=_RUN_DESC)
    ccaa: list[str] | None = Field(default=None, description=_CCAA_DESC)
    horizons: list[Literal[1, 3, 6]] | None = Field(default=None, description=_HORIZONS_DESC)
    as_of_period: str | None = Field(default=None, pattern=r"^\d{4}-(0[1-9]|1[0-2])$", description=_AS_OF_DESC)


_PREDICTIONS_DESC = (
    "Una fila por CCAA y horizonte, con las columnas de la salida original (predictor.py): "
    "anchor_product, context_products, feature_period(_dt) (último mes de dato usado), period(_dt) "
    "(mes objetivo = feature_period + h), ccaa, ccaa_code, model_id, model_name, horizon_months, "
    "y_pred (precio predicho, EUR/kg o litro), y_baseline_calibrated, y_persist_1, y_persist_12, "
    "selected_anchor_features, selected_context_features, selected_exog."
)
_FRESHNESS_DESC = (
    "Diagnóstico de frescura (predictor._compute_data_freshness): main_source_last_period, "
    "main_source_lag_months, exog_last_period, exog_lag_months respecto al momento de la petición."
)


class _PredictResponseBase(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    predictions: list[dict[str, Any]] = Field(..., description=_PREDICTIONS_DESC)
    n_predictions: int
    data_freshness: dict[str, Any] = Field(..., description=_FRESHNESS_DESC)
    warnings: list[str] = Field(
        default_factory=list,
        description="Avisos: desfase MAPA > 24 meses (training.max_validated_lag_months) o CCAA omitidas por histórico insuficiente.",
    )


class PredictBatchResponse(_PredictResponseBase):
    """Batch response."""

    output_path: str | None = None


class PredictInlineResponse(_PredictResponseBase):
    """Inline response."""


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictBatchResponse, PredictInlineResponse]
