"""Pydantic request/response DTOs for the ml15 /train endpoint."""
from pydantic import BaseModel, ConfigDict, Field

from app.application.dto.base_version_dto import VersionInfo


class TrainRequest(BaseModel):
    """Retrain request: a CSV with the 16 feature_columns + target, optionally 'date'."""

    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(
        ...,
        description=(
            "Ruta a un panel mensual con el formato de data/processed/model_ready_panel.csv del "
            "equipo de IA: 'date' (o 'origin_date', o 'year'+'month'), las 16 feature_columns "
            "del artefacto (ipi_national_current, ipi_national_lag_1..6, chem_sector_lag_11, "
            "copper_lag_14, eur_usd_lag_17, oil_brent_lag_12, usa_lag_1, month_sin, month_cos, "
            "quarter, is_spring_risk) y el target ipi_national_t_plus_6 (vacío donde aún no se "
            "conoce). Procedimiento original: test = últimos 13 meses de origen, validación = los "
            "12 anteriores, y se reentrena desde cero un Pipeline(StandardScaler + "
            "Ridge(alpha=25.0)) con las filas cuyo objetivo cae antes de la validación."
        ),
    )
    mlflow_run_id: str = Field(
        default="",
        description=(
            "Run de MLflow opcional: si se indica, el modelo reentrenado también se guarda en él "
            "(trazabilidad). Con o sin run, el modelo reentrenado pasa a ser el modelo base."
        ),
    )
    reference_prices_path: str = Field(
        default="",
        description=(
            "Opcional, junto con reference_financial_path: CSV mensual con el mismo formato que "
            "global_phytosanitary_prices_b2020.csv. Si se indica, pasa a ser el histórico de "
            "referencia del modelo base nuevo (lags del IPI y exógenas al predecir solo con el IPI)."
        ),
    )
    reference_financial_path: str = Field(
        default="",
        description=(
            "Opcional, junto con reference_prices_path: CSV mensual con el mismo formato que "
            "financial_proxies_monthly.csv (cobre, Brent, EUR/USD, IYM)."
        ),
    )


class TrainResponse(BaseModel):
    """Retrain response: test metrics of the original protocol (src/get_stats/get_stats.py).

    Metrics are null when the test window has no row with an observed target (or, for r2, a
    single one) — like the original, which reports evaluation_available=false there.
    """

    model_config = ConfigDict(protected_namespaces=())
    detail: str
    n_train_rows: int = Field(..., description="Filas de entrenamiento (objetivo antes de la validación)")
    n_test_rows: int = Field(..., description="Filas de test con valor real observado (evaluables)")
    train_end_date: str = Field(..., description="Último mes objetivo admitido en entrenamiento (YYYY-MM-01)")
    test_start_date: str
    test_end_date: str
    rmse: float | None
    mae: float | None
    mape_pct: float | None
    r2: float | None
    mda_pct: float | None
    baseline_rmse: float | None = Field(None, description="RMSE de la persistencia (y_pred = IPI actual)")
    baseline_mape_pct: float | None = None
    rmse_reduction_pct: float | None = Field(None, description="Mejora de RMSE frente a la persistencia (%)")
    mape_reduction_pct: float | None = None
    training_time_s: float
    upload_warning: str | None = None
    base_version: VersionInfo | None = None
