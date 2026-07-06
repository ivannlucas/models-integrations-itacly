"""Train DTOs for ml35 dairy ANN fine-tuning."""
from pydantic import BaseModel, ConfigDict, Field


class TrainRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(..., description="Path to CSV with 8 ANN features + consumo_agua_l target")
    mlflow_run_id: str = ""


class TrainResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    detail: str
    mae: float = Field(..., description="Mean absolute error on training data after fine-tuning (L)")
    r2: float = Field(..., description="R² on training data after fine-tuning")
    n_samples: int = Field(..., description="Number of training samples used")
