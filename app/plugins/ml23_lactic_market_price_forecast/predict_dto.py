"""Pydantic DTOs for ml23 (GRU dairy price forecast) /predict endpoint."""
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class PredictBatchRequest(BaseModel):
    """Batch request: CSV with 32 feature columns, grouped by (producto, canal)."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["batch"] = "batch"
    data_path: str = Field(
        ..., description="Ruta al CSV con las 32 features del dataset lácteo"
    )


class PredictBatchResponse(BaseModel):
    """Batch response: one GRU prediction per (producto, canal, fecha) window."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    predictions: list[dict[str, Any]]
    output_path: str | None = None


class PredictInlineRequest(BaseModel):
    """Single-row GRU inference — all 32 feature columns optional (default 0.0)."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["inline"] = "inline"
    model_key: str | None = None
    threshold: float | None = None

    year: float | None = None
    mes: float | None = None
    trimestre: float | None = None
    es_verano: float | None = None
    es_navidad: float | None = None
    precio_lag_1: float | None = None
    precio_lag_3: float | None = None
    precio_lag_12: float | None = None
    variacion_mensual: float | None = None
    media_movil_3: float | None = None
    media_movil_12: float | None = None
    hicp_all_es_lag1: float | None = None
    hicp_dairy_es_lag1: float | None = None
    hicp_energy_es_lag1: float | None = None
    ipc_alimentos_lag1: float | None = None
    ipc_general_lag1: float | None = None
    ipc_vivienda_energia_lag1: float | None = None
    mapa_indper7_cultivos_forrajeros_lag1: float | None = None
    mapa_indper8_ganado_para_abasto_lag1: float | None = None
    mapa_indper8_leche_lag1: float | None = None
    mapa_indper8_vacuno_para_abasto_lag1: float | None = None
    mapa_indpermov10_ganado_para_abasto_lag1: float | None = None
    mapa_indpermov10_leche_lag1: float | None = None
    mapa_indpermov10_vacuno_para_abasto_lag1: float | None = None
    mapa_indpermov9_cereales_lag1: float | None = None
    mapa_indpermov9_cultivos_forrajeros_lag1: float | None = None
    mapa_indpermov9_leguminosas_pienso_lag1: float | None = None
    mapa_preper5_leche_cabra_100_litros_lag1: float | None = None
    mapa_preper5_leche_oveja_100_litros_lag1: float | None = None
    mapa_preper5_leche_vaca_100_litros_lag1: float | None = None
    mapa_preper6_vacas_paridas_100kg_vivo_lag1: float | None = None
    current_price: float | None = None


class PredictInlineResponse(BaseModel):
    """Inline response: GRU price prediction (€/litre) at 6-month horizon."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    prediction: Any = Field(..., description="Precio predicho (€/litro) a 6 meses vista")
    confidence: float | None = None
    horizon: int = Field(default=6, description="Horizonte de predicción en meses")
    features_used: list[str]
    model_version: str = Field(default="1.0.0")
    xai_feature_values: dict[str, float] | None = Field(
        default=None,
        description="Valores de features usados — consumidos por el servicio XAI",
    )


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictBatchResponse, PredictInlineResponse]
