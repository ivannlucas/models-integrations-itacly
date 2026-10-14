"""Train DTOs for ml36 dairy DNL optimizer MLP retraining."""
from pydantic import BaseModel, ConfigDict, Field

from app.application.dto.train_dto import MlflowRunId


class TrainRequest(BaseModel):
    """Retraining request: CSV path with the 8 base features + 2 targets."""

    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(
        ...,
        description=(
            "Path to CSV with T_serv, F_milk, T_in, Fat_perc, Viscosity, Regeneration_perc, "
            "Delta_P, t_ciclo + targets T_out, CO2_emissions. Hydraulic_Resistance, "
            "T_serv_margin and Thermal_Intensity are derived if absent. Split 70/15/15 by "
            "cycle (cycle_id, or rebuilt from t_ciclo resets), balanced by season when a "
            "season/month/date column exists; at least 3 cycles."
        ),
    )
    mlflow_run_id: MlflowRunId


class TrainResponse(BaseModel):
    """Retraining response: per-target regression metrics on the test split (real units)."""

    model_config = ConfigDict(protected_namespaces=())
    detail: str
    mae_t_out: float = Field(..., description="MAE de T_out (°C) en test")
    mse_t_out: float = Field(..., description="MSE de T_out en test")
    r2_t_out: float = Field(..., description="R² de T_out en test")
    mae_co2: float = Field(..., description="MAE de CO2_emissions (kg) en test")
    mse_co2: float = Field(..., description="MSE de CO2_emissions en test")
    r2_co2: float = Field(..., description="R² de CO2_emissions en test")
    n_samples: int = Field(..., description="Filas del CSV")
    n_train: int = Field(..., description="Filas de entrenamiento")
    n_val: int = Field(..., description="Filas de validación (early stopping)")
    n_test: int = Field(..., description="Filas de test (métricas)")
    epochs_executed: int = Field(..., description="Épocas ejecutadas (con early stopping)")
    best_val_loss: float = Field(..., description="Mejor pérdida MSE de validación (escala normalizada)")
    training_time_s: float = Field(..., description="Duración del entrenamiento (s)")
    upload_warning: str | None = Field(
        default=None, description="Aviso si el modelo reentrenado no se ha guardado en MLflow"
    )
