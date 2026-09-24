"""Pydantic request/response DTOs for the ml15 /train endpoint."""
from pydantic import BaseModel, ConfigDict, Field


class TrainRequest(BaseModel):
    """Retrain request: a CSV with the 16 feature_columns + target, optionally 'date'."""

    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(
        ...,
        description=(
            "Ruta a un CSV con las 16 columnas de feature_columns del artefacto "
            "(ipi_national_current, ipi_national_lag_1..6, chem_sector_lag_11, copper_lag_14, "
            "eur_usd_lag_17, oil_brent_lag_12, usa_lag_1, month_sin, month_cos, quarter, "
            "is_spring_risk) más el target ipi_national_t_plus_6, y opcionalmente 'date' para "
            "ordenar cronológicamente el holdout de evaluación. Se reentrena desde cero un "
            "Pipeline(StandardScaler + Ridge(alpha=25.0)) — mismos hiperparámetros que el "
            "equipo de IA (inbox/a15/manifest.yaml), nunca elegidos por el agente."
        ),
    )
    mlflow_run_id: str = ""


class TrainResponse(BaseModel):
    """Retrain response: hold-out metrics on a chronological tail split of the training CSV."""

    model_config = ConfigDict(protected_namespaces=())
    detail: str
    n_train_rows: int
    n_test_rows: int
    rmse: float
    mae: float
    mape_pct: float
    r2: float
    mda_pct: float
    upload_warning: str | None = None
