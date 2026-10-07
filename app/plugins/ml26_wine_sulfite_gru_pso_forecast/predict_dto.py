"""Pydantic request/response DTOs for the ml26 (wine sulfite GRU-PSO, 72 h) /predict endpoint."""
# pylint: disable=too-few-public-methods  # Pydantic DTOs carry no behaviour

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.plugins.ml26_wine_sulfite_gru_pso_forecast.constants import WINDOW

WineType = Literal["red", "rose", "white"]
Stage = Literal["must", "fermentation", "stabilization", "storage", "pre_bottling"]


class LotInput(BaseModel):
    """Fixed data of one tank/lot (one row of lote.csv)."""

    model_config = ConfigDict(protected_namespaces=())
    lot_id: str = Field(..., description="Identificador del lote/depósito")
    wine_type: WineType = Field(..., description="Tipo de vino: red, rose o white")
    volume_l: float = Field(
        ..., gt=0, description="Volumen del depósito en litros (entrenado en 2512–14983 L)"
    )
    ambient_temp_c: float = Field(
        ..., description="Temperatura ambiente de la bodega en °C (entrenado en 10–28 °C)"
    )


class ReadingInput(BaseModel):
    """One 2-hour reading of the lot (one row of lecturas.csv)."""

    model_config = ConfigDict(protected_namespaces=())
    lot_id: Optional[str] = Field(
        default=None, description="Opcional en inline: si falta se usa el del lote"
    )
    timestamp: Optional[str] = Field(
        default=None, description="Marca temporal ISO-8601 de la lectura (paso nominal 2 h)"
    )
    timestamp_index: Optional[int] = Field(
        default=None, description="Índice del paso; si falta se deriva del orden de timestamp"
    )
    stage: Stage = Field(
        ..., description="Etapa: must, fermentation, stabilization, storage, pre_bottling"
    )
    temperature_c: float = Field(..., description="Temperatura del vino (°C)")
    dissolved_oxygen_mg_l: float = Field(..., description="Oxígeno disuelto (mg/L)")
    density_g_ml: float = Field(..., description="Densidad (g/mL)")
    co2_g_l: float = Field(..., description="CO2 disuelto (g/L)")
    ph_lab: Optional[float] = Field(
        default=None, description="pH de laboratorio — solo en filas con analítica"
    )
    free_sulfite_lab: Optional[float] = Field(
        default=None, description="SO2 libre de laboratorio (mg/L) — solo en filas con analítica"
    )
    total_sulfite_lab: Optional[float] = Field(
        default=None, description="SO2 total de laboratorio (mg/L) — solo en filas con analítica"
    )
    dose_mg_l: Optional[float] = Field(
        default=None, description="Dosis de SO2 añadida en este paso (mg/L)"
    )
    stage_progress: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "OBLIGATORIO. Avance de la etapa de esta lectura (0 = inicio, 1 = fin de la etapa). "
            "El modelo auditado se entrenó con este dato (manifest KI-01)."
        ),
    )


class PredictInlineRequest(BaseModel):
    """Inline request: one lot + its reading history (>= 24 readings), or a pre-processed window."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["inline"] = "inline"
    model_key: str | None = None
    threshold: float | None = Field(
        default=None, description="No usado por este modelo (regresión)"
    )
    lot: Optional[LotInput] = Field(
        default=None, description="Datos fijos del depósito (modo operativo)"
    )
    readings: Optional[list[ReadingInput]] = Field(
        default=None,
        description=(
            f"Historial de lecturas del lote, de más antigua a más reciente, mínimo {WINDOW} (48 "
            f"h). Se recomienda "
            "enviar el historial completo del lote: la analítica de laboratorio se arrastra "
            "desde la última muestra."
        ),
    )
    feature_window: Optional[list[list[float]]] = Field(
        default=None,
        description=(
            f"Modo reproducibilidad (memoria §4.2): ventana ya procesada ({WINDOW} x 22) en el "
            f"orden de "
            "best_model.json::feature_columns. Excluyente con lot/readings. Solo para evaluación "
            "técnica."
        ),
    )
    mlflow_run_id: str = Field(
        default="", description="MLflow run ID de un modelo reentrenado por el usuario"
    )

    @model_validator(mode="after")
    def _one_input_mode(self) -> "PredictInlineRequest":
        """Require either (lot + readings) or feature_window, not both."""
        operational = self.lot is not None or self.readings is not None
        if operational and self.feature_window is not None:
            raise ValueError(
                "Use lot+readings (operativo) o feature_window (reproducibilidad), no ambos"
            )
        if self.feature_window is None and (self.lot is None or self.readings is None):
            raise ValueError("Se requiere lot + readings, o feature_window")
        if self.readings is not None and len(self.readings) < WINDOW:
            raise ValueError(
                f"Se requieren al menos {WINDOW} lecturas (48 h); recibidas {len(self.readings)}"
            )
        return self


class LotPrediction(BaseModel):
    """72 h forecast for one lot."""

    model_config = ConfigDict(protected_namespaces=())
    lot_id: Optional[str] = None
    timestamp: Optional[str] = Field(
        default=None, description="Instante de predicción (última lectura usada)"
    )
    timestamp_index: Optional[int] = None
    future_free_sulfite_72h: float = Field(
        ..., description="SO2 libre esperado dentro de 72 h (mg/L)"
    )
    underprotection_risk_72h: float = Field(
        ...,
        description=(
            "Score de riesgo de infraprotección a 72 h en [0, 1] (no es probabilidad " "calibrada)"
        ),
    )
    risk_band: str = Field(
        ..., description="bajo (<0.33) / medio (<0.66) / alto — bandas de la memoria v2.5 §7.1"
    )


class PredictInlineResponse(LotPrediction):
    """Inline response for a single lot."""

    model_id: str
    model_name: str
    model_source: str = Field(
        ..., description="fixed (artefacto del equipo de IA) o mlflow:<run_id>"
    )
    xai_feature_values: dict[str, Any] | None = Field(
        default=None,
        description="Las 22 features del último paso de la ventana — consumido por el servicio XAI",
    )


class PredictBatchRequest(BaseModel):
    """Batch request: CSV of readings with the lot columns repeated on every row."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["batch"] = "batch"
    data_path: str = Field(
        ...,
        description=(
            "CSV (local o s3://) con una fila por lectura: lot_id, wine_type, volume_l, "
            "ambient_temp_c, stage, "
            "temperature_c, dissolved_oxygen_mg_l, density_g_ml, co2_g_l, stage_progress "
            "(obligatorio, sin vacíos) y opcionalmente timestamp, timestamp_index, ph_lab, "
            "free_sulfite_lab, total_sulfite_lab, dose_mg_l. "
            f"Se devuelve una predicción por lote (mínimo {WINDOW} lecturas por lote)."
        ),
    )
    mlflow_run_id: str = Field(
        default="", description="MLflow run ID de un modelo reentrenado por el usuario"
    )


class PredictBatchResponse(BaseModel):
    """Batch response: one forecast per lot."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    model_source: str
    n_lots: int
    predictions: list[LotPrediction]


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictBatchResponse, PredictInlineResponse]
