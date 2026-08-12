"""Pydantic request/response DTOs for the ml3 wine disease/pest /predict endpoint."""
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from app.plugins.ml3_wine_disease_pest_forecast.constants import WINDOW_SIZE


class PredictBatchRequest(BaseModel):
    """Batch request: a CSV/parquet of raw hourly sensor rows (one or more series)."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["batch"] = "batch"
    data_path: str = Field(
        ...,
        description=(
            "Ruta (local o s3://) a un CSV o parquet con las columnas del contrato de entrada "
            "y una fila por hora: Fecha, Temp_Amb_C, Hum_Rel_Pct, Lluvia_mm, Viento_kmh, "
            "CO2_ppm, VOC_ppb, Hum_Suelo_Pct, pH_Suelo y, opcionalmente, ID_Serie. Se devuelve "
            "una predicción por serie (última ventana de 168 horas)."
        ),
    )
    mlflow_run_id: str = Field(default="", description="MLflow run ID for a user-retrained model")


class PredictBatchResponse(BaseModel):
    """Batch response: one scored window per series (last 168-hour window)."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    predictions: list[dict[str, Any]]
    output_path: str | None = None


class PredictInlineRequest(BaseModel):
    """Inline request: at least 168 hourly rows of one series' sensor history."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["inline"] = "inline"
    model_key: str | None = None
    threshold: float | None = None
    rows: list[dict[str, Any]] = Field(
        ...,
        min_length=WINDOW_SIZE,
        description=(
            f"Lista de al menos {WINDOW_SIZE} filas horarias (una por hora), ordenadas de más "
            "antigua a más reciente, con las columnas del contrato de entrada (Fecha, "
            "Temp_Amb_C, Hum_Rel_Pct, Lluvia_mm, Viento_kmh, CO2_ppm, VOC_ppb, Hum_Suelo_Pct, "
            "pH_Suelo y, opcionalmente, ID_Serie). La predicción se calcula sobre la última "
            "ventana de 168 horas. Con menos de 168 filas el código entregado hace padding "
            "repitiendo la primera fila (predicción degradada) — ver manifest known_issues."
        ),
    )
    mlflow_run_id: str = Field(default="", description="MLflow run ID for a user-retrained model")


class PredictInlineResponse(BaseModel):
    """Inline response: ensemble diagnosis, severity, treatment and confidence for one window."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    id_serie: str | int
    fecha_evaluacion: str | None
    diagnostico_ia: str
    confianza_clasificacion: float = Field(
        ..., description="Probabilidad media (0-1) de la clase ganadora tras Soft Voting"
    )
    grado_severidad: float = Field(..., description="Severidad media (0-1) de las 3 regresiones")
    tratamiento_recomendado: str
    probabilidades_clases: dict[str, float] = Field(
        ..., description="Probabilidad media por clase (Soft Voting), clave = clase"
    )
    model_name: str
    xai_feature_values: dict[str, Any] | None = Field(
        default=None, description="Última fila cruda de la ventana predicha — consumido por XAI"
    )


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictBatchResponse, PredictInlineResponse]
