"""Data Transfer Objects (DTOs) for training a model."""
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

# Run de MLflow donde se persiste el modelo reentrenado: obligatorio y no vacío. El modelo
# reentrenado solo vive en MLflow, así que sin run no hay dónde guardarlo (→ 422).
MlflowRunId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class TrainRequest(BaseModel):
    """Request body for training a model.

    Only used by models without their own train_dto (the non-trainable ml28/ml31/ml33):
    mlflow_run_id stays a plain required str so they always answer 501 with the reason,
    even for an empty run id. Trainable plugins declare ``mlflow_run_id: MlflowRunId``.
    """
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
