"""Pydantic request/response DTOs for the ml37 predictive-maintenance /predict endpoint."""
from typing import Annotated, Any, Literal, Union

from pydantic import Field

from app.application.dto.plugin_dto import BatchRequestBase, PluginDTO, UserModelRunIdEs
from app.plugins.ml37_wine_press_predictive_maintenance.constants import PHYSICAL_SENSORS

_SENSORS_DESC = (
    f"Serie temporal de un ciclo: lista de lecturas {{t: segundos, {', '.join(PHYSICAL_SENSORS)}}}"
    f", mínimo 1 lectura, orden arbitrario (se reordena por t). Se remuestrea internamente a "
    "600 pasos equiespaciados — mismo contrato que el adaptador tabular externo del equipo de IA."
)


class PredictInlineRequest(PluginDTO):
    """Inline request: one cycle's raw physical-sensor time series."""

    mode: Literal["inline"] = "inline"
    model_key: str | None = None
    threshold: float | None = Field(
        default=None, description="No usado por este modelo (clasificación multitarea)"
    )
    readings: list[dict[str, float]] = Field(..., description=_SENSORS_DESC)
    mlflow_run_id: UserModelRunIdEs = ""


class PredictBatchRequest(BatchRequestBase):
    """Batch request: a CSV/parquet with one row per (cycle_id, t) reading."""

    data_path: str = Field(
        ...,
        description=(
            "Ruta local o s3:// a un CSV/parquet en formato long: una fila por lectura temporal "
            "de un ciclo, columnas cycle_id, t, " + ", ".join(PHYSICAL_SENSORS) + " — mismo "
            "contrato que src/data_processing/adapt_external_tabular.py del equipo de IA."
        ),
    )


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest],
    Field(discriminator="mode"),
]


class PredictInlineResponse(PluginDTO):
    """Inline response: per-head prediction, uncertainty and neurosymbolic alerts for one cycle."""

    model_id: str
    predictions: dict[str, int] = Field(
        ..., description="pred_<target>: código real UCI por head (cooler/valve/pump/accumulator)"
    )
    risk: dict[str, float] = Field(
        ...,
        description="risk_<target>: incertidumbre predictiva [0,1] = 1 - max(softmax). NO es "
        "riesgo físico del componente.",
    )
    confidence_level: dict[str, str] = Field(
        ..., description="sem_<target>: green/yellow/red (semáforo de incertidumbre)"
    )
    probabilities: dict[str, dict[int, float]] = Field(
        ..., description="Probabilidad softmax de cada clase real por head"
    )
    alerts: list[dict[str, Any]] = Field(
        ...,
        description="Vista fusionada reglas+modelo por componente (igual que alerts_all.parquet "
        "del entregable): component, source (rule|model|combined), severity "
        "(INFO|WARNING|CRITICAL|EMERGENCY), message",
    )
    rule_evidence: list[dict[str, Any]] = Field(
        ...,
        description="TODAS las reglas físicas que se dispararon para este ciclo (puede haber "
        "más de una por componente en accumulator — ver manifest known_issues sobre la vista "
        "'legacy' de alerts, que colapsa a una sola).",
    )


class BatchCyclePrediction(PluginDTO):
    """One cycle's prediction row in a batch response."""

    cycle_id: str
    predictions: dict[str, int] | None = None
    risk: dict[str, float] | None = None
    confidence_level: dict[str, str] | None = None
    alerts: list[dict[str, Any]] | None = None
    error: str | None = None


class PredictBatchResponse(PluginDTO):
    """Batch response: one prediction row per cycle_id in the input table."""

    model_id: str
    predictions: list[BatchCyclePrediction]
    n_rows: int
    n_predictions: int = Field(..., description="Ciclos sin error")
    output_path: str | None = None


PredictResponse = Union[PredictBatchResponse, PredictInlineResponse]
