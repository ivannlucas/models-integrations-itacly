from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class PredictInlineRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["inline"] = "inline"
    image_base64: str = Field(..., description="Base64-encoded image (JPEG, PNG or BMP)")


class PredictBatchRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["batch"] = "batch"
    data_path: str = Field(
        ..., description="Path to a ZIP file containing images inside the container"
    )
    model_id: str = ""  # si se pasa, usa el modelo específico entrenado
    user_id: str = ""   # propietario del modelo específico


PredictRequest = Annotated[
    Union[PredictInlineRequest, PredictBatchRequest],
    Field(discriminator="mode"),
]


class PredictInlineResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    categoria: str
    cereal: str
    confianza_categoria: float
    confianza_cereal: float
    probabilidades_categoria: dict[str, float]
    probabilidades_cereal: dict[str, float]


class PredictBatchResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    predictions: list[dict]
    output_path: str | None = None
    model_train_id: str = ""  # si se pasa, usa el modelo específico entrenado
    user_id: str = ""   # propietario del modelo específico


PredictResponse = Union[PredictInlineResponse, PredictBatchResponse]
