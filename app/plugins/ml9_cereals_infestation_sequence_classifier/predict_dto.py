"""Pydantic request/response DTOs for the ml9 (cereal infestation) /predict endpoint."""
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from app.plugins.ml9_cereals_infestation_sequence_classifier.constants import WINDOW_SIZE


class PredictBatchRequest(BaseModel):
    """Batch request: a CSV of raw hourly telemetry for one or more monitored series."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["batch"] = "batch"
    data_path: str = Field(
        ...,
        description=(
            "Ruta (local o s3://) a un CSV de telemetría horaria cruda con las columnas "
            "sample_id, timestamp, co2_ppm, temp_c, ambient_rh_pct y humidity_grain_pct. Si además "
            f"trae la columna target, se devuelven y_true y las métricas de la evaluación. El "
            f"pipeline construye internamente ventanas de {WINDOW_SIZE} observaciones (stride 12) "
            "por sample_id y devuelve una fila por ventana válida."
        ),
    )
    mlflow_run_id: str = Field(default="", description="MLflow run ID de un modelo reentrenado por el usuario")


class PredictBatchResponse(BaseModel):
    """Batch response: one scored window per row, plus aggregates useful for monitoring."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    n_windows: int = Field(..., description="Ventanas válidas puntuadas")
    n_series: int = Field(..., description="sample_id distintos que produjeron al menos una ventana")
    predictions: list[dict[str, Any]]
    class_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Ventanas predichas por clase — señal de deriva recomendada por la memoria (§10)",
    )
    evaluated_metrics: dict[str, float] | None = Field(
        default=None,
        description=(
            "Solo si el CSV trae la columna target: accuracy, balanced_accuracy, f1_macro, "
            "precision_macro, recall_macro y log_loss sobre las ventanas puntuadas."
        ),
    )
    output_path: str | None = None


class PredictInlineRequest(BaseModel):
    """Inline request: at least WINDOW_SIZE hourly observations of one monitored series."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["inline"] = "inline"
    model_key: str | None = None
    threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Confianza mínima exigida a la clase predicha. No altera la predicción: si la "
            "probabilidad máxima queda por debajo, la respuesta marca low_confidence=true para que "
            "el consumidor decida. El modelo entregado no define ningún umbral de decisión."
        ),
    )
    rows: list[dict[str, Any]] = Field(
        ...,
        min_length=WINDOW_SIZE,
        description=(
            f"Lista de al menos {WINDOW_SIZE} observaciones horarias consecutivas del mismo "
            "sample_id, ordenadas de más antigua a más reciente, con las columnas sample_id, "
            "timestamp, co2_ppm, temp_c, ambient_rh_pct y humidity_grain_pct. Se puntúa la ventana "
            "más reciente del histórico recibido. IMPORTANTE: las variables derivadas (diferencias, "
            "pendientes, medias y desviaciones móviles) se calculan sobre TODAS las filas aportadas, "
            f"así que {WINDOW_SIZE} filas es el mínimo operativo, no el óptimo — aportar el mayor "
            "histórico disponible de la serie reproduce con más fidelidad el comportamiento del "
            "modelo original (ver manifest known_issues sensibilidad_al_historial_aportado)."
        ),
    )
    mlflow_run_id: str = Field(default="", description="MLflow run ID de un modelo reentrenado por el usuario")


class PredictInlineResponse(BaseModel):
    """Inline response: class and per-class probabilities for the most recent window."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    sample_id: str
    window_index: int = Field(..., description="Índice de la ventana puntuada dentro de la serie recibida")
    timestamp_start: str
    timestamp_end: str = Field(..., description="Instante al que se refiere la clase predicha")
    pred_class: int = Field(..., description="0=sano, 1=insectos, 2=moho_critico")
    pred_label: str
    proba_sano: float
    proba_insectos: float
    proba_moho_critico: float
    confidence: float = Field(..., description="Probabilidad de la clase predicha")
    low_confidence: bool = Field(
        default=False,
        description="True si se pidió `threshold` y la confianza queda por debajo",
    )
    n_rows_used: int
    n_windows_available: int = Field(
        ..., description="Ventanas que generó el histórico recibido (se devuelve la más reciente)",
    )
    y_true: int | None = Field(
        default=None, description="Etiqueta real de la ventana, solo si las filas traían `target`",
    )
    model_name: str
    xai_feature_values: dict[str, Any] | None = Field(
        default=None, description="Valores usados en la explicación — consumido por el servicio XAI",
    )


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictBatchResponse, PredictInlineResponse]
