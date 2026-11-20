"""DTOs para las operaciones de predict del plugin Ml10DairyDiseaseVectorDetection."""
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Union

from pydantic import Field

from app.application.dto.plugin_dto import BatchRequestBase, BatchResponseBase, PluginDTO


class PredictBatchRequest(BatchRequestBase):
    """DTO para petición de predicción en modo batch."""


class PredictBatchResponse(BatchResponseBase):
    """DTO para respuesta de predicción en modo batch."""
    mlflow_run_id: str = ""
    model_train_id: str = ""


class PredictInlineRequest(PluginDTO):
    """DTO para petición de predicción en modo inline."""
    mode: Literal["inline"] = "inline"
    image_path: str | None = None     # ruta a imagen en disco (vía shared_data)
    image_base64: str | None = None   # imagen JPG/PNG codificada en base64
    det_conf_thresh: float = 0.2
    cls_conf_thresh: float = 0.5
    mlflow_run_id: str = ""


class PredictInlineResponse(PluginDTO):
    """DTO para respuesta de predicción en modo inline."""
    model_id: str
    prediction: str        # especie dominante (fly | mos | tick | no_vectors | desconocido)
    confidence: float
    vectors_count: int     # vectores reconocidos (sin contar los «desconocido»)
    detections: List[Dict[str, Any]]
    species_summary: Dict[str, int]
    unknown_count: int = Field(
        0, description="Detecciones cuyo recorte no se parece a ninguna mosca, mosquito o "
        "garrapata de entrenamiento (species 'desconocido' en detections)"
    )
    es_desconocido: bool = Field(
        False, description="Hay detecciones pero ninguna es un vector reconocido: prediction "
        "vale 'desconocido'"
    )
    umbral_ood: float | None = Field(
        None, description="Umbral de puntuacion_ood (por detección) a partir del cual el "
        "recorte es 'desconocido'; null si el filtro no está disponible"
    )
    annotated_image: str    # imagen con bboxes dibujadas, base64 JPEG
    heatmap_crops: List[Dict[str, Any]] = Field(default_factory=list)    # ver build_heatmap_crops


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictBatchResponse, PredictInlineResponse]
