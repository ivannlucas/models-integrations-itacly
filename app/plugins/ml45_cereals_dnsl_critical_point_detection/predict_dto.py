"""Transport contract for m45.

The minimum inference unit is a 240-timestep window (not a single row), so both modes take
either a CSV data_path or — for inline — explicit per-sensor time-series arrays (same pattern
already used by m47_dnsl_fallas_maquinaria_pasteurizado for the same reason).
"""
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.plugins.ml45_cereals_dnsl_critical_point_detection.constants import SENSOR_COLUMNS


class PredictBatchRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["batch"] = "batch"
    data_path: str = Field(..., description="Path to CSV with timestamp, sensor columns, and optionally cycle_id/fault_name")
    mlflow_run_id: str = Field(default="", description="MLflow run ID for user-trained model")


class PredictBatchResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    predictions: list[dict[str, Any]]
    output_path: str | None = None


class PredictInlineRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["inline"] = "inline"
    data_path: str | None = Field(default=None, description="Path to CSV (alternative to sending sensor arrays)")
    plenum_temp: list[float] | None = Field(default=None, description="Plenum temperature time-series (°C)")
    exhaust_air_temp: list[float] | None = Field(default=None, description="Exhaust air temperature time-series (°C)")
    exhaust_air_humidity: list[float] | None = Field(default=None, description="Exhaust air humidity time-series (%)")
    static_pressure: list[float] | None = Field(default=None, description="Static pressure time-series (mbar)")
    burner_power: list[float] | None = Field(default=None, description="Burner power time-series (%)")
    fan_speed: list[float] | None = Field(default=None, description="Fan speed time-series (RPM)")
    discharge_frequency: list[float] | None = Field(default=None, description="Discharge frequency time-series (Hz)")
    grain_moisture_in: list[float] | None = Field(default=None, description="Inlet grain moisture time-series (%)")
    ambient_temp: list[float] | None = Field(default=None, description="Ambient temperature time-series (°C)")
    ambient_humidity: list[float] | None = Field(default=None, description="Ambient humidity time-series (%)")
    setpoint_temp: list[float] | None = Field(default=None, description="Setpoint temperature time-series (°C)")
    timestamp: list[str] | None = Field(default=None, description="Timestamp per row — same length as the sensor arrays")
    cycle_id: str | int | None = Field(default=None, description="Cycle identifier (optional)")
    model_key: str | None = None
    threshold: float | None = None
    mlflow_run_id: str = Field(default="", description="MLflow run ID for user-trained model")

    @model_validator(mode="after")
    def _check_sensors_or_path(self):
        if self.data_path is not None:
            return self
        if self.timestamp is None:
            raise ValueError("'timestamp' is required when 'data_path' is not provided")
        for s in SENSOR_COLUMNS:
            if getattr(self, s) is None:
                raise ValueError(f"'{s}' is required when 'data_path' is not provided")
        return self


class PredictInlineResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    window_index: int = 1
    timestamp_init: str
    timestamp_end: str
    predicted_anomaly_class: int
    predicted_anomaly_label: str
    anomaly_probability: float
    decision_threshold: float
    estado_interpretativo: str = Field(..., alias="Estado interpretativo")
    evidencia: str = Field(..., alias="Evidencia")
    probabilidad_de_anomalia: float = Field(..., alias="Probabilidad de anomalia")
    umbral_de_deteccion_de_anomalias: float = Field(..., alias="Umbral de detección de anomalias")
    margen_respecto_al_umbral: float = Field(..., alias="Margen respecto al umbral")
    recomendacion: str = Field(..., alias="Recomendacion")


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictBatchResponse, PredictInlineResponse]
