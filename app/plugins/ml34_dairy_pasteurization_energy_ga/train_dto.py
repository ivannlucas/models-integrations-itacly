"""Train DTOs for ml34 dairy pasteurization MLP fine-tuning."""
from pydantic import BaseModel, ConfigDict, Field


class TrainRequest(BaseModel):
    """Fine-tuning request: CSV path with the 5 features + 2 targets."""

    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(
        ...,
        description=(
            "Path to CSV with T_in_leche, F_flow, T_servicio, t_ciclo, Delta_P "
            "+ targets E_consumo, T_out_leche"
        ),
    )
    mlflow_run_id: str = ""


class TrainResponse(BaseModel):
    """Fine-tuning response: per-target regression metrics on the provided data."""

    model_config = ConfigDict(protected_namespaces=())
    detail: str
    rmse_E_consumo: float = Field(..., description="RMSE de E_consumo (kW)")
    mae_E_consumo: float = Field(..., description="MAE de E_consumo (kW)")
    r2_E_consumo: float = Field(..., description="R² de E_consumo")
    rmse_T_out_leche: float = Field(..., description="RMSE de T_out_leche (°C)")
    mae_T_out_leche: float = Field(..., description="MAE de T_out_leche (°C)")
    r2_T_out_leche: float = Field(..., description="R² de T_out_leche")
    n_samples: int = Field(..., description="Número de muestras usadas en el fine-tuning")
    epochs_executed: int = Field(..., description="Épocas ejecutadas (con early stopping)")
