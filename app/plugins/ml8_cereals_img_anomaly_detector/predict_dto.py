"""Pydantic DTOs for the ml8-cereals-img-anomaly-detector /predict endpoint."""
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class PredictInlineRequest(BaseModel):
    """Inline prediction request: one base64-encoded image."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["inline"] = "inline"
    image_base64: str = Field(..., description="Base64-encoded image (JPEG, PNG or BMP)")
    mlflow_run_id: str = Field(default="", description="MLflow run ID for user-trained model")


class PredictBatchRequest(BaseModel):
    """Batch prediction request: ZIP of images."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["batch"] = "batch"
    data_path: str = Field(
        ..., description="Path to a ZIP file containing images inside the container"
    )
    mlflow_run_id: str = ""


PredictRequest = Annotated[
    Union[PredictInlineRequest, PredictBatchRequest],
    Field(discriminator="mode"),
]


class PredictInlineResponse(BaseModel):
    """Inline prediction: predicted category and cereal with their probabilities."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    categoria: str
    cereal: str
    confianza_categoria: float
    confianza_cereal: float
    probabilidades_categoria: dict[str, float]
    probabilidades_cereal: dict[str, float]


class PredictBatchResponse(BaseModel):
    """Batch prediction: one result (or error) per image."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    predictions: list[dict]
    output_path: str | None = None
    mlflow_run_id: str = ""
    model_train_id: str = ""


PredictResponse = Union[PredictInlineResponse, PredictBatchResponse]
