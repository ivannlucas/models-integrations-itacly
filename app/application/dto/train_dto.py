"""Data Transfer Objects (DTOs) for training a model."""
from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    """Request body for training a model."""
    data_path: str = ""
    mlflow_run_id: str


class TrainResponse(BaseModel):
    """Response body for training a model."""
    detail: str
    metrics: dict = {}
    mlflow_run_id: str = Field(
        default="",
        description=(
            "Run de MLflow al que se subieron los artefactos. La plataforma lo toma como "
            "autoritativo frente al run que pre-creó ella: si aquel falló, este es el que "
            "de verdad tiene el modelo entrenado (train-task-manager.ts, confirmedRunId)."
        ),
    )
    upload_warning: str | None = Field(
        default=None, description="Aviso si el modelo reentrenado no se ha guardado en MLflow"
    )
