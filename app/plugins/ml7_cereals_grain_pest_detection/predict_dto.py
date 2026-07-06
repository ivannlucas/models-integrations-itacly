"""Pydantic request/response DTOs for the grain pest-detection /predict endpoint."""
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class PredictInlineRequest(BaseModel):
    """Inline request: one image as path or base64."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["inline"] = "inline"
    image_path: str | None = None
    image_base64: str | None = None
    threshold: float | None = Field(default=None, description="Override del umbral de confianza")
    mlflow_run_id: str = Field(default="", description="MLflow run ID for user-trained model")


class PredictBatchRequest(BaseModel):
    """Batch request: a directory or ZIP of images inside the container."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["batch"] = "batch"
    data_path: str = Field(..., description="Ruta a un directorio o fichero .zip de imágenes")
    mlflow_run_id: str = Field(default="", description="MLflow run ID for user-trained model")


PredictRequest = Annotated[
    Union[PredictInlineRequest, PredictBatchRequest],
    Field(discriminator="mode"),
]


class PredictInlineResponse(BaseModel):
    """Inline response: dominant species plus all detections and an annotated image."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    prediction: str
    confidence: float
    total_detections: int
    species_counts: dict[str, int]
    detections: list[dict[str, Any]]
    annotated_image: str
    threshold: float | None = None
    features_used: list[str]


class PredictBatchResponse(BaseModel):
    """Batch response: one prediction dict per image."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    predictions: list[dict[str, Any]]
    output_path: str | None = None


PredictResponse = Union[PredictInlineResponse, PredictBatchResponse]
