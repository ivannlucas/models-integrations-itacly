from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field

CLIP_LENGTH = 32


class PredictBatchRequest(BaseModel):
    mode: Literal["batch"] = "batch"
    data_path: str = Field(..., description="Absolute path to a video file (MP4, AVI)")
    detection_threshold: float = Field(default=0.5)
    anomaly_threshold: float = Field(default=0.5)


class PredictBatchResponse(BaseModel):
    model_id: str
    predictions: list[dict[str, Any]]
    output_path: str | None = None


class PredictInlineRequest(BaseModel):
    mode: Literal["inline"] = "inline"
    model_key: str | None = None
    threshold: float | None = None
    frames_base64: list[str] = Field(
        ...,
        min_length=CLIP_LENGTH,
        description=f"List of at least {CLIP_LENGTH} JPEG/PNG frames encoded in base64.",
    )
    detection_threshold: float | None = None


class PredictInlineResponse(BaseModel):
    model_id: str
    threshold: float | None = None
    prediction: Any
    confidence: float | None = None
    features_used: list[str]
    is_anomaly: bool
    behavior_idx: int


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictBatchResponse, PredictInlineResponse]
