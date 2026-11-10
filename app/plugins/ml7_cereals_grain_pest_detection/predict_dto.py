"""Pydantic request/response DTOs for the grain pest-detection /predict endpoint."""
from typing import Annotated, Any, Literal, Union

from pydantic import Field

from app.application.dto.plugin_dto import (
    BatchRequestBase,
    BatchResponseBase,
    PluginDTO,
    UserModelRunId,
)


class PredictInlineRequest(PluginDTO):
    """Inline request: one image as path or base64."""

    mode: Literal["inline"] = "inline"
    image_path: str | None = None
    image_base64: str | None = None
    threshold: float | None = Field(default=None, description="Override del umbral de confianza")
    mlflow_run_id: UserModelRunId = ""


class PredictBatchRequest(BatchRequestBase):
    """Batch request: a directory or ZIP of images inside the container."""

    data_path: str = Field(..., description="Ruta a un directorio o fichero .zip de imágenes")
    mlflow_run_id: UserModelRunId = ""


PredictRequest = Annotated[
    Union[PredictInlineRequest, PredictBatchRequest],
    Field(discriminator="mode"),
]


class PredictInlineResponse(PluginDTO):
    """Inline response: dominant species plus all detections and an annotated image."""

    model_id: str
    prediction: str
    confidence: float
    total_detections: int
    species_counts: dict[str, int]
    detections: list[dict[str, Any]]
    annotated_image: str
    threshold: float | None = None
    features_used: list[str]


class PredictBatchResponse(BatchResponseBase):
    """Batch response: one prediction dict per image."""


PredictResponse = Union[PredictInlineResponse, PredictBatchResponse]
