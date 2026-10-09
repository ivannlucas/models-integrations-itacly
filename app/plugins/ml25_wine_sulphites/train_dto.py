"""DTOs for wine sulphite training operations."""
from pydantic import BaseModel, ConfigDict, Field

from app.application.dto.train_dto import MlflowRunId


class TrainRequest(BaseModel):
    """Request body that specifies the CSV file to use for training."""

    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(..., description="Path to the CSV training file inside the container")
    mlflow_run_id: MlflowRunId


class TrainResponse(BaseModel):
    """Response body returned after a successful training run."""

    model_config = ConfigDict(protected_namespaces=())
    detail: str = Field(..., description="Human-readable result message")
    mae_quality: float = Field(
        ..., description="Mean absolute error of the quality model on the held-out test split"
    )
    mae_bound_so2: float = Field(
        ...,
        description="Mean absolute error of the bound SO2 model on the held-out test split (mg/L)",
    )
    n_train: int = Field(..., description="Number of samples used for training (80 % split)")
    n_test: int = Field(..., description="Number of samples used for evaluation (20 % split)")
    training_time_s: float = Field(..., description="Wall-clock training time in seconds")
    upload_warning: str | None = Field(
        default=None, description="Set if artifacts were saved locally but S3 upload failed"
    )
    mlflow_run_id: str = Field(
        default="",
        description=(
            "Run de MLflow al que se subieron los artefactos. La plataforma lo toma como "
            "autoritativo frente al run que pre-creó ella: si aquel falló, este es el que "
            "de verdad tiene el modelo entrenado (train-task-manager.ts, confirmedRunId)."
        ),
    )
