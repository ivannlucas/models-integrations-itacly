"""Pydantic request/response DTOs for the thermal-mastitis /predict endpoint."""
from typing import Annotated, Any, Literal, Union

from pydantic import Field

from app.application.dto.plugin_dto import BatchResponseBase, PluginDTO


class PredictBatchRequest(PluginDTO):
    """Batch request: a directory or ZIP of thermal images."""

    mode: Literal["batch"] = "batch"
    data_path: str = Field(..., description="Ruta a un directorio o .zip de imágenes térmicas")


class PredictBatchResponse(BatchResponseBase):
    """Batch response: one prediction dict per image."""


class PredictInlineRequest(PluginDTO):
    """Inline request: a single thermal image as base64 or path."""

    mode: Literal["inline"] = "inline"
    model_key: str | None = None
    threshold: float | None = None
    image_base64: str | None = Field(default=None, description="Imagen térmica base64")
    image_path: str | None = Field(default=None, description="Ruta a una imagen térmica en disco")


class PredictInlineResponse(PluginDTO):
    """Inline response: predicted class and per-class probabilities."""

    model_id: str
    threshold: float | None = None
    prediction: Any = Field(..., description="Clase predicha: 'Healthy' o 'SCM'")
    confidence: float | None = Field(None, description="Probabilidad softmax de la clase ganadora")
    features_used: list[str]
    predicted_class_index: int
    probability_healthy: float
    probability_scm: float
    heatmap_url: str | None = Field(
        default=None,
        description="CAM superpuesto en base64 JPEG data URI (misma salida que predict_batch)",
    )


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictBatchResponse, PredictInlineResponse]
