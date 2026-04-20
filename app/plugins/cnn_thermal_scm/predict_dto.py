from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, ConfigDict

CLASS_NAMES = ["Healthy", "SCM"]


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    prediction: str
    predicted_class_index: int
    confidence: float
    probability_healthy: float
    probability_scm: float


class PredictBatchRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["batch"] = "batch"
    data_path: str = Field(
        ..., description="Path to directory with thermal images (JPEG/PNG/BMP)"
    )


class PredictBatchResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    predictions: list[dict[str, Any]]
    output_path: str | None = None


class PredictInlineRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["inline"] = "inline"
    model_key: str | None = None
    threshold: float | None = None
    image_path: str = Field(..., description="Absolute or relative path to a thermal image file")


class PredictInlineResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    threshold: float | None = None
    prediction: Any
    confidence: float | None = None
    features_used: list[str]
    predicted_class_index: int
    probability_healthy: float
    probability_scm: float


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest],
    Field(discriminator="mode"),
]

PredictApiResponse = Union[PredictBatchResponse, PredictInlineResponse]
