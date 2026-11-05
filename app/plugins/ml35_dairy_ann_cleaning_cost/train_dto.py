"""Train DTOs for ml35 dairy ANN fine-tuning."""
from pydantic import BaseModel, ConfigDict, Field

from app.application.dto.train_dto import MlflowRunId


class TrainRequest(BaseModel):
    """Fine-tuning request: CSV with the 8 ANN features plus the consumo_agua_l target."""

    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(
        ..., description="Path to CSV with 8 ANN features + consumo_agua_l target"
    )
    mlflow_run_id: MlflowRunId


class TrainResponse(BaseModel):
    """Fine-tuning result: training-data metrics and the MLflow upload outcome."""

    model_config = ConfigDict(protected_namespaces=())
    detail: str
    mae: float = Field(
        ..., description="Mean absolute error on training data after fine-tuning (L)"
    )
    r2: float = Field(..., description="R² on training data after fine-tuning")
    n_samples: int = Field(..., description="Number of training samples used")
    upload_warning: str | None = Field(
        default=None, description="Aviso si el modelo reentrenado no se ha guardado en MLflow"
    )
