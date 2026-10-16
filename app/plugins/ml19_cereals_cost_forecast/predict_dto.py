"""Pydantic request/response DTOs for the ml19 /predict endpoint.

See inbox/a19/manifest.yaml ("DECISION DE ARQUITECTURA") for why this contract takes a month
label instead of a price history: the model's features depend on a frozen reference dataset
bundled with the plugin, not on client-supplied data.
"""
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

_DATE_DESC = (
    "Mes a consultar dentro del dataset de referencia empaquetado (YYYY-MM). Si se omite, se "
    "usa el mes mas reciente disponible (equivalente a --last del CLI original)."
)


class RegResult(BaseModel):
    """Regression leg of one horizon's prediction."""

    model_name: str = Field(..., description="Ridge | XGBoost")
    pred: float = Field(..., description="Retorno porcentual esperado del precio medio de cereales")
    signal: int = Field(..., description="-1, 0 o 1 (signo de 'pred'; 0 solo si pred==0 exacto)")
    signal_str: str = Field(..., description="LONG | SHORT | FLAT")


class ClfResult(BaseModel):
    """Classification leg of one horizon's prediction."""

    model_name: str = Field(..., description="LogReg | XGBoost")
    prob_up: float = Field(..., description="P(subida) de predict_proba")
    pred_class: int = Field(..., description="0 | 1")
    signal: int = Field(..., description="-1 | 1")
    signal_str: str = Field(..., description="LONG | SHORT")


class HorizonResult(BaseModel):
    """Full result (regression + classification + ensemble) for one horizon."""

    horizon: int = Field(..., description="1, 2 o 3 (meses)")
    predicted_for: str = Field(..., description="Mes calendario que predice este horizonte (YYYY-MM)")
    reg: RegResult | None = None
    clf: ClfResult | None = None
    ensemble_signal: int | None = Field(default=None, description="np.sign(signal_reg + signal_clf)")
    ensemble_str: str | None = Field(default=None, description="LONG | SHORT | FLAT")
    confidence: str | None = Field(
        default=None, description="ALTA (reg y clf coinciden en signo) | BAJA (discrepan)",
    )


class PredictInlineRequest(BaseModel):
    """Inline request: optional month label to query the bundled reference dataset."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["inline"] = "inline"
    date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}$", description=_DATE_DESC)
    mlflow_run_id: str = Field(default="", description="Sin efecto -- ml19 no soporta reentrenamiento")


class PredictInlineResponse(BaseModel):
    """Inline response: multi-horizon ensemble signal for one reference month."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    date_label: str = Field(..., description="Mes 'hoy' usado (fila de referencia), YYYY-MM")
    mapa_month_used: str = Field(
        ..., description="Mes MAPA real que alimenta las features de esa fila (date_label - 3 meses)",
    )
    train_cutoff: str = Field(..., description="training_meta.json['train_end']")
    horizons: list[HorizonResult]


class PredictBatchRequest(BaseModel):
    """Batch request: a CSV with a single 'date' column (YYYY-MM), one row per month to query."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["batch"] = "batch"
    data_path: str = Field(
        ...,
        description=(
            "Ruta local o s3:// a un CSV con una columna 'date' (YYYY-MM): los meses del "
            "dataset de referencia empaquetado que se quieren consultar."
        ),
    )
    mlflow_run_id: str = Field(default="", description="Sin efecto -- ml19 no soporta reentrenamiento")


class BatchPrediction(BaseModel):
    """One requested month's result (or the error if that month isn't in the bundled dataset)."""

    date_requested: str
    date_label: str | None = None
    mapa_month_used: str | None = None
    horizons: list[HorizonResult] | None = None
    error: str | None = None


class PredictBatchResponse(BaseModel):
    """Batch response: one entry per requested month."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    train_cutoff: str
    predictions: list[BatchPrediction]
    n_rows: int
    n_predictions: int = Field(..., description="Filas sin error (fecha encontrada en el dataset)")
    output_path: str | None = None


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictBatchResponse, PredictInlineResponse]
