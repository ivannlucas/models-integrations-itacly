"""Pydantic request/response DTOs for ml39's /predict endpoint — Integrated Gradients (always)
and SHAP (opt-in) per-cycle explanations of ml37's predictions."""
from typing import Annotated, Literal, Union

from pydantic import Field

from app.application.dto.plugin_dto import BatchRequestBase, PluginDTO, UserModelRunIdEs
from app.plugins.ml39_wine_predictive_maintenance_xai.constants import PHYSICAL_SENSORS

_SENSORS_DESC = (
    f"Serie temporal de un ciclo: lista de lecturas {{t: segundos, {', '.join(PHYSICAL_SENSORS)}}}"
    f", mínimo 1 lectura, orden arbitrario (se reordena por t). Mismo contrato que "
    "ml37-wine-press-predictive-maintenance — a39 explica el mismo ciclo que a37 predice."
)
_SHAP_DESC = (
    "Añade importancia SHAP por sensor y por head, usando el fondo persistido "
    "(shap_background.npy). Coste alto en CPU (varios segundos por ciclo); por defecto false."
)


class PredictInlineRequest(PluginDTO):
    """Inline request: one cycle's raw physical-sensor time series (same contract as a37)."""

    mode: Literal["inline"] = "inline"
    model_key: str | None = None
    threshold: float | None = Field(
        default=None, description="No usado por este modelo (capa de explicabilidad)"
    )
    readings: list[dict[str, float]] = Field(..., description=_SENSORS_DESC)
    include_shap: bool = Field(default=False, description=_SHAP_DESC)
    mlflow_run_id: UserModelRunIdEs = ""


class PredictBatchRequest(BatchRequestBase):
    """Batch request: a CSV/parquet with one row per (cycle_id, t) reading (same contract as
    a37's predict_batch)."""

    data_path: str = Field(
        ...,
        description=(
            "Ruta local o s3:// a un CSV/parquet en formato long: una fila por lectura temporal "
            "de un ciclo, columnas cycle_id, t, " + ", ".join(PHYSICAL_SENSORS) + "."
        ),
    )
    include_shap: bool = Field(default=False, description=_SHAP_DESC)


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest],
    Field(discriminator="mode"),
]


class PredictInlineResponse(PluginDTO):
    """Inline response: per-head Integrated Gradients (and optional SHAP) sensor importance for
    the class that ml37 would predict for this same cycle."""

    model_id: str
    predicted_class: dict[str, int] = Field(
        ...,
        description="Código real UCI que IG/SHAP explican por head — la misma predicción que "
        "devolvería ml37 para este ciclo, incluida aquí para que la respuesta sea autocontenida.",
    )
    ig_importance: dict[str, dict[str, float]] = Field(
        ...,
        description="head -> sensor -> |IG| medio sobre los 600 timesteps, para la clase "
        "predicha de ese head (perfil 'full': steps=50).",
    )
    shap_importance: dict[str, dict[str, float]] | None = Field(
        default=None,
        description="head -> sensor -> |SHAP| medio, solo si include_shap=true; null en caso "
        "contrario.",
    )


class BatchCycleExplanation(PluginDTO):
    """One cycle's explanation row in a batch response."""

    cycle_id: str
    predicted_class: dict[str, int] | None = None
    ig_importance: dict[str, dict[str, float]] | None = None
    shap_importance: dict[str, dict[str, float]] | None = None
    error: str | None = None


class PredictBatchResponse(PluginDTO):
    """Batch response: one explanation row per cycle_id in the input table."""

    model_id: str
    predictions: list[BatchCycleExplanation]
    n_rows: int
    n_predictions: int = Field(..., description="Ciclos sin error")
    output_path: str | None = None


PredictResponse = Union[PredictBatchResponse, PredictInlineResponse]
