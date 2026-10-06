from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PredictBatchRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["batch"] = "batch"
    data_path: str = Field(..., description="Path to CSV file with sensor time-series data")
    mlflow_run_id: str = Field(default="", description="MLflow run ID for user-trained model")


class PredictBatchResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    predictions: list[dict[str, Any]]
    output_path: str | None = None


class PredictInlineRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["inline"] = "inline"
    data_path: str | None = Field(default=None, description="Path to CSV file (alternative to sending sensor arrays)")
    PS1: list[float] | None = Field(default=None, description="Pressure sensor 1 time-series (bar)")
    PS3: list[float] | None = Field(default=None, description="Pressure sensor 3 time-series (bar)")
    EPS1: list[float] | None = Field(default=None, description="Motor power time-series (W)")
    FS1: list[float] | None = Field(default=None, description="Flow rate time-series (L/min)")
    TS1: list[float] | None = Field(default=None, description="Temperature in time-series (°C)")
    TS2: list[float] | None = Field(default=None, description="Temperature out time-series (°C)")
    VS1: list[float] | None = Field(default=None, description="Vibration time-series (mm/s)")
    Time_Segundos: list[float] | None = Field(default=None, description="Time in seconds for each observation")
    Cycle_ID: int | None = Field(default=None, description="Cycle identifier")
    mlflow_run_id: str = Field(default="", description="MLflow run ID for user-trained model")

    @model_validator(mode="after")
    def _check_sensors_or_path(self):
        if self.data_path is not None:
            return self
        for s in ["PS1", "PS3", "EPS1", "FS1", "TS1", "TS2", "VS1"]:
            if getattr(self, s) is None:
                raise ValueError(f"'{s}' is required when 'data_path' is not provided")
        return self


class PredictInlineResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    Enfriador_Fouling: int = Field(..., description="0=SANO, 1=WARNING, 2=CRÍTICO")
    Valvula_Switch: int = Field(..., description="0=SANO, 1=WARNING, 2=CRÍTICO")
    Bomba_Leakage: int = Field(..., description="0=SANO, 1=WARNING, 2=CRÍTICO")
    Acumulador_Gas: int = Field(..., description="0=SANO, 1=WARNING, 2=CRÍTICO")
    Confianza_Fouling: float = Field(..., description="Confidence for Fouling prediction (0-1)")
    Confianza_Valvula: float = Field(..., description="Confidence for Valve prediction (0-1)")
    Confianza_Bomba: float = Field(..., description="Confidence for Pump prediction (0-1)")
    Confianza_Acumulador: float = Field(..., description="Confidence for Accumulator prediction (0-1)")
    model_name: str


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictBatchResponse, PredictInlineResponse]
