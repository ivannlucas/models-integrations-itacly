"""Pydantic request/response DTOs for the ml46 (DNSL fouling/clog) /predict endpoint."""
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from app.plugins.ml46_dairy_fouling_clog_detection.constants import SEQ_LEN


class PredictBatchRequest(BaseModel):
    """Batch request: a CSV of raw telemetry rows (one or more assets, ordered by timestamp)."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["batch"] = "batch"
    data_path: str = Field(
        ...,
        description=(
            "Ruta a un CSV de telemetría cruda con, como mínimo, las columnas del contrato "
            "de entrada (timestamp, asset_id, flow_kg_s, dP_kPa, ...). El pipeline construye "
            "ventanas de 120 minutos internamente y devuelve una fila por ventana válida."
        ),
    )
    mlflow_run_id: str = Field(default="", description="MLflow run ID for user fine-tuned model")


class PredictBatchResponse(BaseModel):
    """Batch response: one scored window per row, plus consolidated alert episodes."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    predictions: list[dict[str, Any]]
    alerts: list[dict[str, Any]] = Field(default_factory=list)
    output_path: str | None = None


class PredictInlineRequest(BaseModel):
    """Inline request: at least SEQ_LEN minutes of raw telemetry history for one asset."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["inline"] = "inline"
    model_key: str | None = None
    threshold: float | None = Field(
        default=None,
        description="Override opcional de policy['watch_foul_prob_thr'] (por defecto el calibrado del artefacto).",
    )
    rows: list[dict[str, Any]] = Field(
        ...,
        min_length=SEQ_LEN,
        description=(
            f"Lista de al menos {SEQ_LEN} filas de telemetría cruda (una por minuto), ordenadas "
            "de más antigua a más reciente, con las columnas del contrato de entrada (timestamp, "
            "asset_id, flow_kg_s, dP_kPa, vibration_mm_s, temperaturas, setpoints, composición "
            "nominal, last_maintenance_type...). La predicción se calcula sobre la ventana de 120 "
            "minutos que termina en la última fila. Para activos que NO estaban en el conjunto de "
            "entrenamiento (todo activo de producción real), se recomienda aportar el mayor "
            "histórico disponible del activo (idealmente >= 8h) para que el baseline de residuos "
            "se calcule con precisión — con solo 120 filas, las predicciones siguen siendo válidas "
            "pero pueden desviarse unos puntos porcentuales del valor exacto del checkpoint."
        ),
    )
    mlflow_run_id: str = Field(default="", description="MLflow run ID for user fine-tuned model")


class PredictInlineResponse(BaseModel):
    """Inline response: model outputs for the last window plus operator explanation."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    asset_id: str
    timestamp: str
    pred_severity: float
    pred_stage: int
    pred_stage_name: str
    p_stage0: float
    p_stage1: float
    p_stage2: float
    p_foul_h: float = Field(..., description="Probabilidad de inicio de ensuciamiento en 30 min")
    p_actionable_foul_h: float = Field(..., description="Probabilidad de ensuciamiento no planificado en 120 min")
    p_clog_h: float = Field(..., description="Probabilidad de inicio de obstrucción en 15 min")
    pred_tte_foul_min: float
    pred_tte_clog_min: float
    pred_ttu_min: float
    operator_status: str
    priority: str
    recommended_action: str
    activated_predicates: str
    is_alert: bool = Field(..., description="True si esta ventana genera un episodio de alerta accionable")
    model_name: str
    xai_feature_values: dict[str, Any] | None = Field(
        default=None, description="Valores de features usados — consumido por el servicio XAI",
    )


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictBatchResponse, PredictInlineResponse]
