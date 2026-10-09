"""Pydantic DTOs for the ml14 /train endpoint."""
# pylint: disable=too-few-public-methods  # Pydantic DTOs carry no behaviour
from pydantic import BaseModel, ConfigDict, Field

from app.application.dto.train_dto import MlflowRunId


class TrainRequest(BaseModel):
    """Refit the deployed GRU with the AI team's procedure on a weekly dataset."""

    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(
        ...,
        description=(
            "Ruta local o s3:// a un CSV semanal (domingos) en uno de estos formatos: el dataset de "
            "modelado del equipo de IA (FECHA + las 39 features, como final_dataset_for_modeling.csv) "
            "o las series en bruto que acepta /predict (date, PROTECCION_FITO, CARBURANTES, "
            "COPPER_EUR_TON, GAS_EUR_MMBTU, COIL_EUR_BARRIL, DEXUSEU). Split temporal 80/20."
        ),
    )
    mlflow_run_id: MlflowRunId = Field(
        ...,
        description="Run de MLflow donde se persiste el modelo reentrenado. Obligatorio: el "
        "artefacto base nunca se sobrescribe.",
    )


class TrainResponse(BaseModel):
    """GRU test metrics as in models/metrics/model_comparison.json (mean ± std of 3 seeds)."""

    model_config = ConfigDict(protected_namespaces=())
    detail: str
    mae: float
    mae_std: float
    rmse: float
    rmse_std: float
    mape_pct: float
    mape_pct_std: float
    r2: float
    r2_std: float
    direction_acc_pct: float = Field(..., description="Seed 42, frente al precio actual (Naive)")
    skill_score: float = Field(..., description="Seed 42: 1 - MSE_modelo / MSE_Naive")
    drift_rmse: float = Field(..., description="RMSE del baseline Drift en el mismo test")
    naive_rmse: float
    beats_drift: bool = Field(..., description="Criterio de aceptación del equipo de IA: RMSE GRU < RMSE Drift")
    n_train: int
    n_test: int
    epochs_run_seed_42: int
    n_seeds: int
    training_time_s: float
    mlflow_run_id: str
    upload_warning: str | None = Field(
        default=None,
        description="Compatibilidad con la plataforma: siempre None (un fallo de subida es un 502)",
    )
