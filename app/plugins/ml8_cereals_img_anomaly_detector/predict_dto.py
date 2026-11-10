"""Pydantic DTOs for the ml8-cereals-img-anomaly-detector /predict endpoint."""
from typing import Annotated, Literal, Union

from pydantic import Field

from app.application.dto.plugin_dto import (
    BatchRequestBase,
    BatchResponseBase,
    PluginDTO,
    UserModelRunId,
)


class PredictInlineRequest(PluginDTO):
    """Inline prediction request: one base64-encoded image."""

    mode: Literal["inline"] = "inline"
    image_base64: str = Field(..., description="Base64-encoded image (JPEG, PNG or BMP)")
    mlflow_run_id: UserModelRunId = ""


class PredictBatchRequest(BatchRequestBase):
    """Batch prediction request: ZIP of images."""

    data_path: str = Field(
        ..., description="Path to a ZIP file containing images inside the container"
    )


PredictRequest = Annotated[
    Union[PredictInlineRequest, PredictBatchRequest],
    Field(discriminator="mode"),
]


class PredictInlineResponse(PluginDTO):
    """Inline prediction: predicted category and cereal with their probabilities."""

    model_id: str
    categoria: str
    cereal: str
    confianza_categoria: float
    confianza_cereal: float
    probabilidades_categoria: dict[str, float]
    probabilidades_cereal: dict[str, float]


class PredictBatchResponse(BatchResponseBase):
    """Batch prediction: one result (or error) per image."""

    predictions: list[dict]
    mlflow_run_id: str = ""
    model_train_id: str = ""


PredictResponse = Union[PredictInlineResponse, PredictBatchResponse]
