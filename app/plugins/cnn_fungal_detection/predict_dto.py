from typing import Annotated, Any, Dict, List, Literal, Union

from pydantic import BaseModel, Field


class PredictBatchRequest(BaseModel):
    mode: Literal["batch"] = "batch"
    data_path: str = Field(..., description="Path to directory or .zip with images inside the container")


class PredictBatchResponse(BaseModel):
    model_id: str
    predictions: List[Dict[str, Any]]
    output_path: str | None = None


class PredictInlineRequest(BaseModel):
    mode: Literal["inline"] = "inline"
    image_path: str = Field(..., description="Absolute path to a JPEG/PNG/BMP image file")
    threshold: float | None = None


class PredictInlineResponse(BaseModel):
    model_id: str
    threshold: float | None = None
    prediction: str
    confidence: float
    features_used: List[str]
    probabilities: Dict[str, float]


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictBatchResponse, PredictInlineResponse]
