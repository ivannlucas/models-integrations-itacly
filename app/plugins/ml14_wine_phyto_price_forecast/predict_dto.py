"""Pydantic request/response DTOs for the ml14 /predict endpoint."""
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from app.plugins.ml14_wine_phyto_price_forecast.constants import MIN_HISTORY_ROWS


class PredictBatchRequest(BaseModel):
    """Batch request: a CSV with a raw weekly history (date + 6 base columns).

    Mirrors the delivered CLI (`python -m src.main predict --input <csv>`): the whole CSV is
    one continuous weekly history and yields a single prediction anchored at its last row —
    there is no sliding-window multi-prediction mode in the delivered code (see
    inbox/a14/manifest.yaml known_issues).
    """

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["batch"] = "batch"
    data_path: str = Field(
        ...,
        description=(
            "Ruta a un CSV con histórico semanal (W-SUN) continuo: date, PROTECCION_FITO, "
            "CARBURANTES, COPPER_EUR_TON, GAS_EUR_MMBTU, COIL_EUR_BARRIL, DEXUSEU. "
            f"Mínimo {MIN_HISTORY_ROWS} filas consecutivas sin huecos ni duplicados."
        ),
    )


class PredictBatchResponse(BaseModel):
    """Batch response: a single prediction anchored at the CSV's last row."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    predictions: list[dict[str, Any]]
    n_predictions: int
    output_path: str | None = None


class PredictInlineRequest(BaseModel):
    """Inline request: the raw weekly history as a list of row dicts."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["inline"] = "inline"
    model_key: str | None = None
    threshold: float | None = Field(
        default=None,
        description="No usado por este modelo (no hay umbral de decisión); se acepta por compatibilidad de contrato.",
    )
    rows: list[dict[str, Any]] = Field(
        ...,
        min_length=MIN_HISTORY_ROWS,
        description=(
            "Histórico semanal (W-SUN) continuo, sin huecos ni fechas duplicadas: date, "
            "PROTECCION_FITO, CARBURANTES, COPPER_EUR_TON, GAS_EUR_MMBTU, COIL_EUR_BARRIL, "
            f"DEXUSEU. Mínimo {MIN_HISTORY_ROWS} filas. Se predice el precio a 16 semanas "
            "desde la última fila del histórico."
        ),
    )


class PredictInlineResponse(BaseModel):
    """Inline response: LSTM price forecast at the 16-week horizon."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    predicted_price: float = Field(..., description="PROTECCION_FITO predicho a horizon_weeks vista (índice base 2020=100).")
    current_price: float = Field(..., description="PROTECCION_FITO de la última fila del histórico aportado.")
    drift_baseline: float = Field(..., description="Baseline lineal de drift — referencia frente a la que el LSTM no consigue mejorar en test final (ver manifest).")
    horizon_weeks: int
    last_observed_date: str
    prediction_date: str
    model_used: str = Field(default="LSTM")
    gap_warning: str | None = Field(
        default=None,
        description="Aviso si la última fecha del histórico aportado está muy por detrás de hoy.",
    )
    n_rows_used: int
    xai_feature_values: dict[str, float] | None = Field(
        default=None, description="Valores de la última fila de features usados — consumidos por el servicio de explicabilidad",
    )


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictBatchResponse, PredictInlineResponse]
