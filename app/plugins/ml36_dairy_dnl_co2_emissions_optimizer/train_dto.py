"""Train DTOs for ml36 dairy DNL optimizer MLP fine-tuning."""
from pydantic import BaseModel, ConfigDict, Field


class TrainRequest(BaseModel):
    """Fine-tuning request: CSV path with the 8 base features + 2 targets."""

    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(
        ...,
        description=(
            "Path to CSV with T_serv, F_milk, T_in, Fat_perc, Viscosity, Regeneration_perc, "
            "Delta_P, t_ciclo + targets T_out, CO2_emissions. Hydraulic_Resistance, "
            "T_serv_margin and Thermal_Intensity are derived if absent; an optional "
            "cycle_id column keeps whole cycles together in the validation hold-out."
        ),
    )
    mlflow_run_id: str = ""


class TrainResponse(BaseModel):
    """Fine-tuning response: per-target regression metrics on the provided data."""

    model_config = ConfigDict(protected_namespaces=())
    detail: str
    mae_t_out: float = Field(..., description="MAE de T_out (°C)")
    mse_t_out: float = Field(..., description="MSE de T_out")
    r2_t_out: float = Field(..., description="R² de T_out")
    mae_co2: float = Field(..., description="MAE de CO2_emissions (kg)")
    mse_co2: float = Field(..., description="MSE de CO2_emissions")
    r2_co2: float = Field(..., description="R² de CO2_emissions")
    n_samples: int = Field(..., description="Número de muestras usadas en el fine-tuning")
    epochs_executed: int = Field(..., description="Épocas ejecutadas (con early stopping)")
    upload_warning: str | None = Field(
        default=None, description="Aviso si el modelo reentrenado no se ha guardado en MLflow"
    )
