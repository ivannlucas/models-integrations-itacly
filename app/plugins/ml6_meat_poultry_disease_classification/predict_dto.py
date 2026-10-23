"""Pydantic request/response DTOs for the ml6 poultry disease /predict endpoint."""
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

_MODEL_KEY_DESCRIPTION = (
    "Ensemble a usar: 'thermal' (imagen térmica del ave, healthy/sick) o 'rgb' (imagen RGB de "
    "heces, healthy/cocci/ncd/salmo). Obligatorio: las dos modalidades no son intercambiables."
)

_MLFLOW_RUN_DESCRIPTION = (
    "Run de MLflow con un ensemble reentrenado por el usuario (/train). Vacío = modelo base. Si el "
    "run no tiene un ensemble cargable de esa modalidad, la petición falla (422): nunca se usa el "
    "base en su lugar."
)
ModelKey = Literal["thermal", "rgb"]


class PredictBatchRequest(BaseModel):
    """Batch request: a directory or ZIP of images of one modality."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["batch"] = "batch"
    model_key: ModelKey | None = Field(default=None, description=_MODEL_KEY_DESCRIPTION)
    mlflow_run_id: str = Field(default="", description=_MLFLOW_RUN_DESCRIPTION)
    data_path: str = Field(
        ...,
        description=(
            "Ruta local, s3:// o .zip con imágenes. Si las imágenes están en subcarpetas con el "
            "nombre de su clase (thermal: healthy/sick; rgb: healthy|pcrhealthy, cocci|pcrcocci, "
            "ncd|pcrncd, salmo|pcrsalmo) se devuelven además las métricas de evaluación."
        ),
    )


class PredictBatchResponse(BaseModel):
    """Batch response: one prediction row per image, plus metrics when labels are known."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    task: str
    n_images: int
    predictions: list[dict[str, Any]]
    evaluated_metrics: dict[str, Any] | None = Field(
        default=None,
        description=("Métricas por modelo y del ensemble si todas las imágenes traen su "
                     "clase real"),
    )
    output_path: str | None = None


class PredictInlineRequest(BaseModel):
    """Inline request: a single image as base64 or path."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["inline"] = "inline"
    model_key: ModelKey | None = Field(default=None, description=_MODEL_KEY_DESCRIPTION)
    mlflow_run_id: str = Field(default="", description=_MLFLOW_RUN_DESCRIPTION)
    threshold: float | None = None
    image_base64: str | None = Field(default=None, description="Imagen codificada en base64")
    image_path: str | None = Field(default=None, description="Ruta a una imagen en disco")


class PredictInlineResponse(BaseModel):
    """Inline response: majority-vote class plus each base model's vote and probabilities."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    task: str
    threshold: float | None = None
    prediction: str = Field(
        ..., description="Clase predicha por el ensemble (votación por mayoría)",
    )
    predicted_class_index: int
    confidence: float = Field(
        ..., description="Fracción de modelos base que votan la clase predicha (1/3, 2/3 o 1)",
    )
    model_predictions: dict[str, str] = Field(..., description="Clase votada por cada modelo base")
    model_probabilities: dict[str, dict[str, float]] = Field(
        ..., description="Softmax de cada modelo base (informativo: la decisión es por voto)",
    )
    features_used: list[str]
    warnings: list[str] = Field(default_factory=list)


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictBatchResponse, PredictInlineResponse]
