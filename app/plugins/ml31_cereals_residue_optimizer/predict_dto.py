"""Pydantic request/response DTOs for the cereal residue-optimizer /predict endpoint."""
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class PredictBatchRequest(BaseModel):
    """Batch request: a CSV of cereal-scenario rows."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["batch"] = "batch"
    data_path: str = Field(..., description="Path to CSV with cereal scenario features")


class PredictBatchResponse(BaseModel):
    """Batch response: one prediction dict per row."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    predictions: list[dict[str, Any]]
    output_path: str | None = None


class PredictInlineRequest(BaseModel):
    """Inline request: a single cereal scenario."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["inline"] = "inline"
    model_key: str | None = None
    threshold: float | None = None
    Sup_Secano_ha: float = Field(..., description="Superficie en secano (ha)")
    Sup_Regadio_ha: float = Field(..., description="Superficie en regadío (ha)")
    Lluvia_Primavera_mm: float = Field(..., description="Precipitación primaveral (mm)")
    Sequia_Primavera: int = Field(default=0, description="1 si lluvia < 200mm (sequía), 0 si no")
    Cultivo: str = Field(..., description="Tipo de cultivo: Trigo, Cebada, Maíz, Girasol, etc.")


class PredictInlineResponse(BaseModel):
    """Inline response: predicted available soil residue (tons)."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    prediction: Any = Field(..., description="Residuo disponible predicho (toneladas)")
    confidence: float | None = None
    xai_feature_values: dict[str, Any] | None = Field(
        default=None, description="Valores de features usados — consumido por el servicio XAI",
    )


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictBatchResponse, PredictInlineResponse]
