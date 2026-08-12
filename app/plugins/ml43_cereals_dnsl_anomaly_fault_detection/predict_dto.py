from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class PredictBatchRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["batch"] = "batch"
    data_path: str = Field(..., description="Path to CSV file with sensor time-series data (one or more cycles)")
    mlflow_run_id: str = Field(default="", description="MLflow run ID for a user-trained model")


class PredictBatchResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    predictions: list[dict[str, Any]]
    output_path: str | None = None


class PredictInlineRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["inline"] = "inline"
    temp_zona1: float = Field(..., description="Temperatura zona 1 del horno (°C)")
    temp_zona2: float = Field(..., description="Temperatura zona 2 del horno (°C)")
    temp_zona3: float = Field(..., description="Temperatura zona 3 del horno (°C)")
    temp_salida_gases: float = Field(..., description="Temperatura de salida de gases (°C)")
    presion_camara: float = Field(..., description="Presión interna de la cámara (mbar)")
    presion_ventilacion: float = Field(..., description="Presión del sistema de ventilación (mbar)")
    potencia_kw: float = Field(..., description="Potencia eléctrica consumida (kW)")
    flujo_gas: float = Field(..., description="Flujo de gas del combustible (m3/h)")
    humedad_relativa: float = Field(..., description="Humedad relativa interior (%)")
    temp_ambiente: float = Field(..., description="Temperatura ambiente exterior (°C)")
    setpoint_temp: float = Field(..., description="Temperatura objetivo programada (°C)")
    posicion_valvula: float = Field(..., description="Posición de la válvula de gas (%)")
    velocidad_ventilador: float = Field(..., description="Velocidad del ventilador de circulación (RPM)")
    model_key: str | None = None
    threshold: float | None = Field(default=None, description="Overrides the default decision threshold (0.41)")
    mlflow_run_id: str = Field(default="", description="MLflow run ID for a user-trained model")


class PredictInlineResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    predicted_anomaly_class: int = Field(..., description="0=No Fallo, 1=Fallo")
    predicted_anomaly_label: str = Field(..., description="'Fallo' o 'No Fallo'")
    anomaly_probability: float = Field(..., description="Probabilidad de anomalía [0, 1]")
    decision_threshold: float
    xai_feature_values: dict[str, float]
    corrective_actions: dict | None = None
    xai_error: str | None = None
    model_name: str


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictBatchResponse, PredictInlineResponse]
