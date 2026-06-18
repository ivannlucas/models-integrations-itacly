"""Pydantic request/response DTOs for the cow-behaviour ``/predict`` endpoint."""
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from app.plugins.ml5_meat_cow_behaviour.constants import CLIP_LENGTH


class PredictBatchRequest(BaseModel):
    """Batch request: path to a video file processed frame by frame."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["batch"] = "batch"
    data_path: str = Field(
        ...,
        description=(
            "Ruta absoluta a un vídeo (MP4, AVI) dentro del contenedor. El pipeline "
            "ejecuta Detectron2 + ByteTrack + SlowFast sobre cada frame."
        ),
    )


class PredictBatchResponse(BaseModel):
    """Batch response: per-frame detections with tracked behaviours."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    predictions: list[dict[str, Any]]
    output_path: str | None = None


class PredictInlineRequest(BaseModel):
    """Inline request: a pre-cropped clip of one cow as base64 frames."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["inline"] = "inline"
    model_key: str | None = None
    threshold: float | None = Field(
        default=None,
        description=(
            "Override del umbral de anomalía (por defecto 0.5). Las predicciones con "
            "confianza por debajo de este valor se marcan como anomalía."
        ),
    )
    frames_base64: list[str] = Field(
        ...,
        min_length=CLIP_LENGTH,
        description=(
            f"Lista de al menos {CLIP_LENGTH} frames JPEG/PNG en base64. Cada frame debe "
            "ser el ROI recortado de una sola vaca (224×224 px recomendado). SlowFast "
            "muestrea 8 frames 'slow' + 32 'fast' de este clip."
        ),
    )


class PredictInlineResponse(BaseModel):
    """Inline response: predicted behaviour plus per-class probabilities for XAI."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    threshold: float | None = None
    prediction: Any = Field(..., description="Comportamiento predicho, p.ej. 'grazing'")
    confidence: float | None = Field(
        None, description="Probabilidad softmax del comportamiento predicho (0.0–1.0)"
    )
    features_used: list[str]
    is_anomaly: bool = Field(
        ..., description="True si confidence < threshold de anomalía"
    )
    behavior_idx: int = Field(..., description="Índice numérico de la clase predicha")
    xai_feature_values: dict[str, float] | None = Field(
        default=None,
        description="Probabilidad softmax de cada comportamiento — usado por el servicio XAI",
    )


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictBatchResponse, PredictInlineResponse]
