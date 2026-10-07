"""Pydantic request/response DTOs for the ml15 /predict endpoint.

PredictInlineRequest declares each of the artifact's 16 feature_columns as its own top-level
field (Swagger-documented, typed) — PredictModelUseCase.execute() builds the plugin's
``features`` dict via ``request.model_dump(exclude={"mode", "model_key", "threshold",
"mlflow_run_id", "data_path"})``, so a generic nested ``features: dict`` field here would end up
double-wrapped (``{"features": {...}}``) instead of the flat dict plugin.py expects — this is
the same convention already used by every other plugin in the repo (see e.g.
ml17_meat_market_price_analysis/predict_dto.py).

All 16 feature fields are Optional: any of them can be omitted and derived automatically from
'date'/'origin_date' plus the bundled reference history (see history.py) — mirrors the AI
team's original --input ipi_history.csv mode, which only required date + the current national
IPI value. See preprocessing.py::build_feature_row for the exact derivation/override rules.
"""
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

_CALENDAR_NOTE = "Opcional: se deriva de 'date'/'origin_date' si se omite."
_HISTORY_NOTE = (
    "Opcional: si se omite, se deriva de 'date'/'origin_date' y el histórico de referencia "
    "bundled (ver known_issues del manifest)."
)


class PredictBatchRequest(BaseModel):
    """Batch request: a CSV with one row per prediction."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["batch"] = "batch"
    data_path: str = Field(
        ...,
        description=(
            "Ruta a un CSV con una fila por predicción. Dos formatos aceptados: (a) panel "
            "técnico completo, con las 16 columnas del contrato (ipi_national_current, "
            "ipi_national_lag_1..6, chem_sector_lag_11, copper_lag_14, eur_usd_lag_17, "
            "oil_brent_lag_12, usa_lag_1, más 'date' o las 4 columnas de calendario); o (b) "
            "histórico simple mensual del IPI nacional -- columnas date(/origin_date, o "
            "year+month) + ipi_national_current (o 'spain'/'ipi'/'value'/'y') -- igual que "
            "data/input/ipi_history.csv del equipo de IA; el resto de features se derivan "
            "automáticamente del histórico de referencia bundled, fusionando primero todas "
            "las filas del CSV para que unas sirvan de contexto de retardo a las otras."
        ),
    )
    mlflow_run_id: str = Field(default="", description="MLflow run ID de un modelo reentrenado por el usuario")


class PredictBatchResponse(BaseModel):
    """Batch response: one row per input row (y_pred, or 'error' if that row failed)."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    predictions: list[dict[str, Any]] = Field(
        ...,
        description="Una entrada por fila: row, origin_date/target_date (si derivables), y_pred, y_anchor.",
    )
    n_predictions: int
    output_path: str | None = None


class PredictInlineRequest(BaseModel):
    """Inline request: the artifact's 16 feature_columns, one field each."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["inline"] = "inline"
    model_key: str | None = None
    threshold: float | None = Field(
        default=None, description="No usado por este modelo; se acepta por compatibilidad de contrato."
    )

    date: str | None = Field(
        default=None,
        description="Fecha de origen (YYYY-MM-DD). Alias de 'origin_date'. " + _CALENDAR_NOTE,
    )
    origin_date: str | None = Field(default=None, description="Alias de 'date'.")

    ipi_national_current: float | None = Field(
        default=None,
        description=(
            "Último IPI fitosanitario nacional observado en origin_date (índice base "
            "2020=100). " + _HISTORY_NOTE + " Si se aporta junto con 'date', también "
            "sobrescribe el histórico de referencia en esa fecha (útil para fechas más "
            "recientes de lo que cubre el snapshot bundled)."
        ),
    )
    ipi_national_lag_1: float | None = Field(default=None, description="IPI nacional 1 mes antes de origin_date. " + _HISTORY_NOTE)
    ipi_national_lag_2: float | None = Field(default=None, description="IPI nacional 2 meses antes de origin_date. " + _HISTORY_NOTE)
    ipi_national_lag_3: float | None = Field(default=None, description="IPI nacional 3 meses antes de origin_date. " + _HISTORY_NOTE)
    ipi_national_lag_4: float | None = Field(default=None, description="IPI nacional 4 meses antes de origin_date. " + _HISTORY_NOTE)
    ipi_national_lag_5: float | None = Field(default=None, description="IPI nacional 5 meses antes de origin_date. " + _HISTORY_NOTE)
    ipi_national_lag_6: float | None = Field(default=None, description="IPI nacional 6 meses antes de origin_date. " + _HISTORY_NOTE)
    chem_sector_lag_11: float | None = Field(default=None, description="Índice sectorial químico retardado 11 meses. " + _HISTORY_NOTE)
    copper_lag_14: float | None = Field(default=None, description="Precio del cobre retardado 14 meses. " + _HISTORY_NOTE)
    eur_usd_lag_17: float | None = Field(default=None, description="Tipo de cambio EUR/USD retardado 17 meses. " + _HISTORY_NOTE)
    oil_brent_lag_12: float | None = Field(default=None, description="Precio del petróleo Brent retardado 12 meses. " + _HISTORY_NOTE)
    usa_lag_1: float | None = Field(default=None, description="Índice de pesticidas de EE.UU. retardado 1 mes. " + _HISTORY_NOTE)

    month_sin: float | None = Field(default=None, description="sin(2*pi*mes/12). " + _CALENDAR_NOTE)
    month_cos: float | None = Field(default=None, description="cos(2*pi*mes/12). " + _CALENDAR_NOTE)
    quarter: int | None = Field(default=None, description="Trimestre natural (1-4). " + _CALENDAR_NOTE)
    is_spring_risk: int | None = Field(
        default=None,
        description="1 si abril/mayo/junio (riesgo fúngico primaveral vid), 0 en otro caso. " + _CALENDAR_NOTE,
    )

    mlflow_run_id: str = Field(default="", description="MLflow run ID de un modelo reentrenado por el usuario")


class PredictInlineResponse(BaseModel):
    """Inline response: t+6 national IPI forecast for the given feature vector."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    origin_date: str | None = Field(
        default=None, description="Fecha de origen (YYYY-MM-01), si se pudo derivar de 'date'/'origin_date'."
    )
    target_date: str | None = Field(
        default=None, description="origin_date + horizon (YYYY-MM-01), si origin_date es derivable."
    )
    horizon: int = Field(default=6, description="Horizonte productivo vigente (siempre 6).")
    y_pred: float = Field(..., description="IPI fitosanitario nacional predicho a t+6 (índice base 2020=100).")
    y_anchor: float = Field(..., description="ipi_national_current usado como ancla (referencia de dirección).")
    model_name: str
    xai_feature_values: dict[str, float] | None = Field(
        default=None, description="Valores de las 16 features usadas — consumido por el servicio de explicabilidad."
    )


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictBatchResponse, PredictInlineResponse]
