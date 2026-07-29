"""Pydantic request/response DTOs for the thermal-mastitis /predict endpoint."""
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class PredictBatchRequest(BaseModel):
    """Batch request: a directory or ZIP of thermal images."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["batch"] = "batch"
    data_path: str = Field(..., description="Ruta a un directorio o .zip de imágenes térmicas")


class PredictBatchResponse(BaseModel):
    """Batch response: one prediction dict per image."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    predictions: list[dict[str, Any]]
    output_path: str | None = None


class PredictInlineRequest(BaseModel):
    """Inline request: a single thermal image as base64 or path."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["inline"] = "inline"
    model_key: str | None = None
    threshold: float | None = None
    image_base64: str | None = Field(default=None, description="Imagen térmica base64")
    image_path: str | None = Field(default=None, description="Ruta a una imagen térmica en disco")


class PredictInlineResponse(BaseModel):
    """Inline response: predicted class and per-class probabilities."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    threshold: float | None = None
    prediction: Any = Field(..., description="Clase predicha: 'Healthy' o 'SCM'")
    confidence: float | None = Field(None, description="Probabilidad softmax de la clase ganadora")
    features_used: list[str]
    predicted_class_index: int
    probability_healthy: float
    probability_scm: float


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictBatchResponse, PredictInlineResponse]
