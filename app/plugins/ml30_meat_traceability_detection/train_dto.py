"""Pydantic DTOs for the meat-traceability /train endpoint."""
from pydantic import BaseModel, ConfigDict, Field


class TrainRequest(BaseModel):
    """Train request: a CSV with the 33 features plus the target column."""

    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(..., description="CSV con features + columna target")
    mlflow_run_id: str


class TrainResponse(BaseModel):
    """Train response: fields match the dict returned by plugin.train()."""

    model_config = ConfigDict(protected_namespaces=())
    detail: str
    accuracy: float
    f1: float
    roc_auc: float
    n_train: int
    n_test: int
    training_time_s: float
    upload_warning: str | None = Field(
        default=None,
        description="Informativo si los artefactos se guardaron en local pero falló el upload a S3",
    )
