"""Pydantic request/response DTOs for the ml40 /predict endpoint."""
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class PredictBatchRequest(BaseModel):
    """Batch request: a CSV with one or more full cycles (run_id) of one subsystem."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["batch"] = "batch"
    data_path: str = Field(
        ...,
        description=(
            "Ruta a un CSV de un único subsistema (refrigeracion o aireado — se detecta por "
            "columnas). Admite dos formatos: sensores crudos (run_id, time_min y las columnas "
            "del contrato de entrada; se aplica la ingeniería de variables completa) o datos ya "
            "procesados estilo data/splits/{system}_test.csv (se infiere directamente). El "
            "diagnóstico se emite por ciclo (voto por mayoría sobre run_id)."
        ),
    )
    mlflow_run_id: str = Field(default="", description="MLflow run ID for a user-retrained model")


class PredictBatchResponse(BaseModel):
    """Batch response: one consolidated diagnosis per cycle."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    system: str = Field(..., description="Subsistema detectado: refrigeracion o aireado")
    predictions: list[dict[str, Any]] = Field(
        ..., description="Una fila por run_id: prediction, prediction_name, confidence (0-1) y, "
                         "si la entrada venía etiquetada, fault_id/fault para auditoría."
    )
    n_runs: int
    avg_confidence: float = Field(..., description="Confianza media (0-1) del lote")
    model_health: str = Field(
        ..., description="ESTABLE/DEGRADADO según la confianza media de esta petición vs. 75% "
                         "(adaptación stateless del monitor original)."
    )
    output_path: str | None = None


class PredictInlineRequest(BaseModel):
    """Inline request: the full time series of one cycle as a list of row dicts."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["inline"] = "inline"
    model_key: str | None = None
    system: Optional[Literal["refrigeracion", "aireado"]] = Field(
        default=None,
        description="Subsistema; si se omite se detecta por las columnas de las filas.",
    )
    threshold: float | None = Field(
        default=None,
        description="No usado por este modelo (los umbrales neurosimbólicos vienen calibrados "
                    "en los artefactos); se acepta por compatibilidad de contrato.",
    )
    rows: list[dict[str, Any]] = Field(
        ...,
        min_length=60,
        description=(
            "Serie temporal de UN ciclo: una fila por minuto con las columnas crudas del "
            "subsistema (refrigeracion: T_amb, T_set, T_cab, ..., P_dis_bar; aireado: "
            "Kg_embutido, N_fan_Hz, RH_cab, ...). time_min es obligatorio; si falta run_id se "
            "asume un único ciclo. Histórico mínimo: 100 min (refrigeracion) / 60 min (aireado)."
        ),
    )
    mlflow_run_id: str = Field(default="", description="MLflow run ID for a user-retrained model")


class PredictInlineResponse(BaseModel):
    """Inline response: consolidated diagnosis for the submitted cycle."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    system: str
    run_id: int
    prediction: int = Field(..., description="Clase final tras RF + reglas físicas + voto por ciclo")
    prediction_name: str
    confidence: float = Field(..., description="Confianza media del ciclo (0-1)")
    n_rows_used: int = Field(..., description="Filas realmente puntuadas tras la ingeniería de variables")
    model_health: str
    model_name: str
    xai_feature_values: dict[str, Any] | None = Field(
        default=None, description="Valores agregados usados — consumido por el servicio XAI",
    )


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictBatchResponse, PredictInlineResponse]
