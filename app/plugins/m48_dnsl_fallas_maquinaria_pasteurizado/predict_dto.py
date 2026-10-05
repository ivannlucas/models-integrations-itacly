from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

_DT_DESC = (
    "Desplaza TS1/TS2 al baseline de 65 °C (gemelo digital). Solo para datos del banco UCI de laboratorio; "
    "con datos reales de planta debe ser false. Si se omite se usa APPLY_DIGITAL_TWIN (por defecto false)."
)


class PredictBatchRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["batch"] = "batch"
    data_path: str = Field(..., description="Path to CSV file with sensor time-series data (varios ciclos por Cycle_ID)")
    mlflow_run_id: str = Field(default="", description="MLflow run ID for user-trained model")
    apply_digital_twin: bool | None = Field(default=None, description=_DT_DESC)
    include_xai: bool = Field(
        default=False,
        description="Añade el análisis global de puntos críticos de control (CCP, Grad-CAM promedio) sobre los ciclos del CSV.",
    )
    include_shap: bool = Field(
        default=False,
        description="Con include_xai, calcula además SHAP (GradientExplainer). Coste alto en CPU (minutos).",
    )
    n_samples: int = Field(default=50, gt=0, description="Ciclos máximos del CSV usados en el análisis CCP (muestra aleatoria, seed 42)")

    @model_validator(mode="after")
    def _shap_requires_xai(self):
        if self.include_shap and not self.include_xai:
            raise ValueError("'include_shap' requires 'include_xai'=true")
        return self


class PredictBatchResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    predictions: list[dict[str, Any]]
    output_path: str | None = None
    xai: dict[str, Any] | None = Field(
        default=None,
        description="{ccps: [...], shap: [...], summary: {...}} cuando include_xai=true. 'shap' vacío si no se pidió SHAP.",
    )


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
    apply_digital_twin: bool | None = Field(default=None, description=_DT_DESC)
    include_xai: bool = Field(
        default=True,
        description="Informe local por componente: ventana crítica Grad-CAM, riesgo y acción prescriptiva.",
    )
    include_shap: bool = Field(
        default=False,
        description="Con include_xai, añade SHAP (sensor más relevante) usando el fondo empaquetado. Coste alto en CPU.",
    )
    include_cam: bool = Field(default=False, description="Devuelve además la curva Grad-CAM (600 valores) por componente")

    @model_validator(mode="after")
    def _check_request(self):
        if self.include_shap and not self.include_xai:
            raise ValueError("'include_shap' requires 'include_xai'=true")
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
    xai: list[dict[str, Any]] | None = Field(
        default=None,
        description="Informe local por componente (Fouling, Válvula, Bomba, Acumulador) cuando include_xai=true.",
    )


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictBatchResponse, PredictInlineResponse]
