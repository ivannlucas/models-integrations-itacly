"""DTOs para las operaciones de predict del plugin Modelo10Lacteo."""
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Union

from pydantic import BaseModel, Field, ConfigDict


class PredictBatchRequest(BaseModel):
    """DTO para petición de predicción en modo batch."""
    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["batch"] = "batch"
    data_path: str  # ruta a CSV, ZIP o directorio dentro del contenedor


class PredictBatchResponse(BaseModel):
    """DTO para respuesta de predicción en modo batch."""
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    predictions: List[Dict[str, Any]]
    output_path: str | None = None


class PredictInlineRequest(BaseModel):
    """DTO para petición de predicción en modo inline."""
    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["inline"] = "inline"
    image_path: str | None = None     # ruta a imagen en disco (vía shared_data)
    image_base64: str | None = None   # imagen JPG/PNG codificada en base64
    det_conf_thresh: float = 0.2      # umbral de confianza de detección
    cls_conf_thresh: float = 0.5      # umbral de confianza de clasificación


class PredictInlineResponse(BaseModel):
    """DTO para respuesta de predicción en modo inline."""
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    prediction: str        # especie dominante (fly | mos | tick | no_vectors)
    confidence: float
    vectors_count: int
    detections: List[Dict[str, Any]]
    species_summary: Dict[str, int]


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictBatchResponse, PredictInlineResponse]
