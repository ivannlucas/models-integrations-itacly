"""Pydantic DTOs for the ml14 /train endpoint."""
# pylint: disable=too-few-public-methods  # Pydantic DTOs carry no behaviour
from pydantic import BaseModel, ConfigDict, Field

from app.application.dto.base_version_dto import VersionInfo


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
    mlflow_run_id: str = Field(
        default="",
        description=(
            "Run de MLflow opcional: si se indica, el modelo reentrenado también se guarda en él "
            "(trazabilidad). Con o sin run, el modelo reentrenado pasa a ser el modelo base."
        ),
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
    mlflow_run_id: str = ""
    upload_warning: str | None = Field(
        default=None,
        description=(
            "Aviso si la subida a MLflow falló; el modelo base se publica igualmente "
            "(solo un fallo al publicar la base es un 502)"
        ),
    )
    base_version: VersionInfo | None = Field(
        default=None, description="Versión del modelo base publicada por este reentreno"
    )
