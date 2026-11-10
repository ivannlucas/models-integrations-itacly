"""Pydantic request/response DTOs for the fungal leaf-disease CNN ``/predict`` endpoint."""
from typing import Annotated, Literal, Union

from pydantic import Field

from app.application.dto.plugin_dto import (
    BatchRequestBase,
    BatchResponseBase,
    PluginDTO,
    UserModelRunId,
)


class PredictInlineRequest(PluginDTO):
    """Inline request: a single image encoded as base64."""

    mode: Literal["inline"] = "inline"
    image_base64: str = Field(..., description="Base64-encoded image (JPEG, PNG or BMP)")
    mlflow_run_id: UserModelRunId = ""


class PredictBatchRequest(BatchRequestBase):
    """Batch request: a path to a ZIP file of images inside the container."""

    data_path: str = Field(
        ..., description="Path to a ZIP file containing images inside the container"
    )
    mlflow_run_id: UserModelRunId = ""


PredictRequest = Annotated[
    Union[PredictInlineRequest, PredictBatchRequest],
    Field(discriminator="mode"),
]


class PredictInlineResponse(PluginDTO):
    """Inline response: predicted disease class and per-class probabilities."""

    model_id: str
    prediction: str
    confidence: float
    probabilities: dict[str, float]


class PredictBatchResponse(BatchResponseBase):
    """Batch response: one prediction dict per image in the ZIP."""

    predictions: list[dict]


PredictResponse = Union[PredictInlineResponse, PredictBatchResponse]
