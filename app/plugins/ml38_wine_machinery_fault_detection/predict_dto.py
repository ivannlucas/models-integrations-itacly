"""Pydantic DTOs for the ml38 wine machinery fault-detection /predict endpoint.

Three modes (discriminated by ``mode``):
  - "inline": the 24 features of one window (the tabular contract of the delivered pipeline,
    data/processed/feature_dataset.parquet).
  - "raw":    a vibration CSV + a current CSV of the same test; the service computes the 24
    features like scripts/predict_raw.py.
  - "batch":  CSV or parquet with the 24 feature columns, one prediction per row.
"""
from typing import Annotated, Literal, Union

from pydantic import Field

from app.application.dto.plugin_dto import (
    BatchRequestBase,
    BatchResponseBase,
    InlineRequestBase,
    PluginDTO,
    UnusedThreshold,
    UserModelRunIdEs,
    XaiFeatureValues,
)


class PredictBatchRequest(BatchRequestBase):
    """Batch prediction: a table with the 24 feature columns (family/severity/group_id ignored)."""

    data_path: str = Field(
        ...,
        description=(
            "Ruta (local o s3://) a un CSV o parquet con las 24 columnas de características de "
            "data/processed/feature_dataset.parquet (vib_* y ele_*). Las columnas family, "
            "severity y group_id, si vienen, se ignoran."
        ),
    )


class PredictBatchResponse(BatchResponseBase):
    """Batch response: one prediction (or ``error``) per row, with ``row`` = row index."""


class PredictInlineRequest(InlineRequestBase):
    """One window described by its 24 features (vibration ch1 in g, current phase 1 in A)."""

    threshold: UnusedThreshold = None
    mlflow_run_id: UserModelRunIdEs = ""

    vib_mean: float = Field(..., description="Media de la señal de vibración (g)")
    vib_std: float = Field(..., description="Desviación típica de la vibración (g)")
    vib_max: float = Field(..., description="Máximo de la vibración (g)")
    vib_min: float = Field(..., description="Mínimo de la vibración (g)")
    vib_rms: float = Field(..., description="Valor RMS de la vibración (g)")
    vib_kurtosis: float = Field(..., description="Curtosis (exceso, insesgada) de la vibración")
    vib_skewness: float = Field(..., description="Asimetría (insesgada) de la vibración")
    vib_spectral_energy: float = Field(
        ...,
        description="Energía espectral de la vibración: suma de |FFT|² (depende de la longitud)",
    )
    vib_spectral_centroid: float = Field(
        ..., description="Centroide espectral de la vibración, en índice de bin FFT"
    )
    vib_band_low: float = Field(..., description="Fracción de energía de vibración en 0-200 Hz")
    vib_band_mid: float = Field(..., description="Fracción de energía de vibración en 200-1000 Hz")
    vib_band_high: float = Field(
        ..., description="Fracción de energía de vibración en 1000-5000 Hz"
    )
    vib_amp_1x: float = Field(..., description="Amplitud |FFT| en la frecuencia de giro (1X)")
    vib_amp_2x: float = Field(..., description="Amplitud |FFT| en el segundo armónico (2X)")
    vib_ratio_2x_1x: float = Field(..., description="Cociente de amplitudes 2X/1X")
    ele_mean: float = Field(..., description="Media de la corriente de fase (A)")
    ele_std: float = Field(..., description="Desviación típica de la corriente (A)")
    ele_max: float = Field(..., description="Máximo de la corriente (A)")
    ele_min: float = Field(..., description="Mínimo de la corriente (A)")
    ele_rms: float = Field(..., description="Valor RMS de la corriente (A)")
    ele_kurtosis: float = Field(..., description="Curtosis (exceso, insesgada) de la corriente")
    ele_skewness: float = Field(..., description="Asimetría (insesgada) de la corriente")
    ele_spectral_energy: float = Field(
        ...,
        description="Energía espectral de la corriente: suma de |FFT|² (depende de la longitud)",
    )
    ele_spectral_centroid: float = Field(
        ..., description="Centroide espectral de la corriente, en índice de bin FFT"
    )


class PredictRawRequest(PluginDTO):
    """Raw inference from a vibration/current CSV pair — scripts/predict_raw.py."""

    mode: Literal["raw"] = "raw"
    # model_key="raw" tells predict_inline to compute the features from the signals first.
    model_key: str = "raw"
    threshold: UnusedThreshold = None
    mlflow_run_id: UserModelRunIdEs = ""

    vibration_csv: str = Field(
        ...,
        description=(
            "Ruta (local o s3://) al CSV raw de vibración: columna time + columnas de señal (g). "
            "El modelo se entrenó con el canal 1 (rodamiento lado no accionado del motor, "
            "horizontal) y ventanas de 240000 muestras (12 s a 20 kHz)."
        ),
    )
    current_csv: str = Field(
        ...,
        description=(
            "Ruta (local o s3://) al CSV raw eléctrico del mismo ensayo: columna time + columnas "
            "de señal. Entrenado con corriente de fase 1 (A) y ventanas de 300000 muestras."
        ),
    )
    motor: Literal["Motor-2", "Motor-4"] = Field(
        ..., description="Configuración operativa NLN-EMP de la señal (no selecciona sensores)"
    )
    speed: int = Field(
        ...,
        description=(
            "Velocidad de operación en %: Motor-2 → 50, 75 o 100; Motor-4 → 70. Fija la "
            "frecuencia de giro de las características 1X/2X."
        ),
    )
    window_index: int = Field(
        default=0, ge=0,
        description="Columna de señal posterior a time que se evalúa (0 = la primera)",
    )


class PredictInlineResponse(PluginDTO):
    """Prediction for one window: fault family and, when applicable, binary severity."""

    model_id: str
    family: str = Field(
        ...,
        description=(
            "Familia de fallo: alignment, bearing, cavitation, electrical, healthy, "
            "mechanical_other o unbalance"
        ),
    )
    severity: int | None = Field(
        default=None,
        description="0 = leve (nivel 1), 1 = significativo (nivel > 1); null si no aplica",
    )
    severity_label: str = Field(
        ..., description="leve, significativo, sin fallo (healthy) o modelo no disponible"
    )
    severity_confidence: float | None = Field(
        default=None,
        description=(
            "Probabilidad de severidad significativa del MLP de la familia (null si no aplica)"
        ),
    )
    family_probabilities: dict[str, float] = Field(
        default_factory=dict,
        description="Probabilidad del Random Forest para cada familia (la predicha es la máxima)",
    )
    input_warnings: list[str] = Field(
        default_factory=list,
        description=(
            "Avisos de dominio de la entrada (p. ej., señal raw de longitud distinta a la de "
            "entrenamiento)"
        ),
    )
    xai_feature_values: XaiFeatureValues = None


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest, PredictRawRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictBatchResponse, PredictInlineResponse]
