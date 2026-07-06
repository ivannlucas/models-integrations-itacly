"""Pydantic request/response DTOs for the fungal leaf-disease CNN ``/predict`` endpoint."""
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class PredictInlineRequest(BaseModel):
    """Inline request: a single image encoded as base64."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["inline"] = "inline"
    image_base64: str = Field(..., description="Base64-encoded image (JPEG, PNG or BMP)")
    mlflow_run_id: str = Field(default="", description="MLflow run ID for user-trained model")


class PredictBatchRequest(BaseModel):
    """Batch request: a path to a ZIP file of images inside the container."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["batch"] = "batch"
    data_path: str = Field(
        ..., description="Path to a ZIP file containing images inside the container"
    )
    mlflow_run_id: str = Field(default="", description="MLflow run ID for user-trained model")


PredictRequest = Annotated[
    Union[PredictInlineRequest, PredictBatchRequest],
    Field(discriminator="mode"),
]


class PredictInlineResponse(BaseModel):
    """Inline response: predicted disease class and per-class probabilities."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    prediction: str
    confidence: float
    probabilities: dict[str, float]


class PredictBatchResponse(BaseModel):
    """Batch response: one prediction dict per image in the ZIP."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    predictions: list[dict]
    output_path: str | None = None


PredictResponse = Union[PredictInlineResponse, PredictBatchResponse]
