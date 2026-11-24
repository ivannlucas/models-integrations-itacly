"""Pydantic request/response DTOs for the ml37 predictive-maintenance /predict endpoint.

Includes the per-cycle explainability of deliverable a39 (Integrated Gradients + SHAP, see
xai.py): ``include_xai`` / ``include_shap`` in the request, ``ig_importance`` /
``shap_importance`` / ``xai_error`` in the response.
"""
from typing import Annotated, Any, Literal, Union

from pydantic import Field, model_validator

from app.application.dto.plugin_dto import BatchRequestBase, PluginDTO, UserModelRunIdEs
from app.plugins.ml37_wine_press_predictive_maintenance.constants import (
    IG_STEPS,
    PHYSICAL_SENSORS,
    SHAP_BATCH_MAX_CYCLES_DEFAULT,
)

_SENSORS_DESC = (
    f"Serie temporal de un ciclo: lista de lecturas {{t: segundos, {', '.join(PHYSICAL_SENSORS)}}}"
    f", mínimo 1 lectura, orden arbitrario (se reordena por t). Se remuestrea internamente a "
    "600 pasos equiespaciados — mismo contrato que el adaptador tabular externo del equipo de IA."
)
_XAI_INLINE_DESC = (
    "Explicabilidad del ciclo (entregable a39): importancia por sensor y por head con "
    f"Integrated Gradients (steps={IG_STEPS}). Añade ≈3 s en CPU. Por defecto true."
)
_XAI_BATCH_DESC = (
    "Explicabilidad por ciclo con Integrated Gradients (entregable a39). Por defecto true. "
    "Añade ≈3 s por ciclo en CPU: para lotes grandes usa /predict?wait=false o desactívala."
)
_SHAP_DESC = (
    "Con include_xai, añade la importancia SHAP (GradientExplainer) por sensor y por head. Coste "
    "alto en CPU: ≈70 s por ciclo. Por defecto false."
)
_SHAP_MAX_DESC = (
    "Ciclos máximos con SHAP cuando include_shap=true (los primeros del fichero; el resto lleva "
    "solo Integrated Gradients)."
)
_IG_IMPORTANCE_DESC = (
    "head -> sensor -> |IG| medio sobre los 600 pasos (perfil 'full' del entregable a39: "
    "steps=50, baseline 0, clase de máxima confianza en cada paso). null si include_xai=false "
    "o si la explicación falló (ver xai_error)."
)
_SHAP_IMPORTANCE_DESC = (
    "head -> sensor -> |SHAP| medio sobre los 600 pasos y las clases. null salvo include_shap."
)
_XAI_ERROR_DESC = (
    "Motivo si la explicabilidad del ciclo falló (la predicción sigue siendo válida); null si "
    "fue correcta o no se pidió."
)


class _XaiRequestMixin(PluginDTO):
    """include_shap only makes sense on top of include_xai (same rule as ml48)."""

    @model_validator(mode="after")
    def _shap_requires_xai(self):
        if getattr(self, "include_shap", False) and not getattr(self, "include_xai", False):
            raise ValueError("'include_shap' requires 'include_xai'=true")
        return self


class PredictInlineRequest(_XaiRequestMixin):
    """Inline request: one cycle's raw physical-sensor time series."""

    mode: Literal["inline"] = "inline"
    model_key: str | None = None
    threshold: float | None = Field(
        default=None, description="No usado por este modelo (clasificación multitarea)"
    )
    readings: list[dict[str, float]] = Field(..., description=_SENSORS_DESC)
    mlflow_run_id: UserModelRunIdEs = ""
    include_xai: bool = Field(default=True, description=_XAI_INLINE_DESC)
    include_shap: bool = Field(default=False, description=_SHAP_DESC)


class PredictBatchRequest(BatchRequestBase, _XaiRequestMixin):
    """Batch request: a CSV/parquet with one row per (cycle_id, t) reading."""

    data_path: str = Field(
        ...,
        description=(
            "Ruta local o s3:// a un CSV/parquet en formato long: una fila por lectura temporal "
            "de un ciclo, columnas cycle_id, t, " + ", ".join(PHYSICAL_SENSORS) + " — mismo "
            "contrato que src/data_processing/adapt_external_tabular.py del equipo de IA."
        ),
    )
    include_xai: bool = Field(default=True, description=_XAI_BATCH_DESC)
    include_shap: bool = Field(default=False, description=_SHAP_DESC)
    n_samples: int = Field(default=SHAP_BATCH_MAX_CYCLES_DEFAULT, gt=0, description=_SHAP_MAX_DESC)


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
    ig_importance: dict[str, dict[str, float]] | None = Field(
        default=None, description=_IG_IMPORTANCE_DESC
    )
    shap_importance: dict[str, dict[str, float]] | None = Field(
        default=None, description=_SHAP_IMPORTANCE_DESC
    )
    xai_error: str | None = Field(default=None, description=_XAI_ERROR_DESC)


class BatchCyclePrediction(PluginDTO):
    """One cycle's prediction row in a batch response."""

    cycle_id: str
    predictions: dict[str, int] | None = None
    risk: dict[str, float] | None = None
    confidence_level: dict[str, str] | None = None
    alerts: list[dict[str, Any]] | None = None
    ig_importance: dict[str, dict[str, float]] | None = Field(
        default=None, description=_IG_IMPORTANCE_DESC
    )
    shap_importance: dict[str, dict[str, float]] | None = Field(
        default=None, description=_SHAP_IMPORTANCE_DESC
    )
    xai_error: str | None = Field(default=None, description=_XAI_ERROR_DESC)
    error: str | None = None


class PredictBatchResponse(PluginDTO):
    """Batch response: one prediction row per cycle_id in the input table."""

    model_id: str
    predictions: list[BatchCyclePrediction]
    n_rows: int
    n_predictions: int = Field(..., description="Ciclos sin error")
    output_path: str | None = None


PredictResponse = Union[PredictBatchResponse, PredictInlineResponse]
