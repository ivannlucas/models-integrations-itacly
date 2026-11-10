"""Pydantic DTOs for the meat-traceability /train endpoint."""
from pydantic import Field

from app.application.dto.plugin_dto import ConfirmedRunId, PluginDTO, TrainRequestBase


class TrainRequest(TrainRequestBase):
    """Train request: a CSV with the 33 features plus the target column."""

    data_path: str = Field(..., description="CSV con features + columna target")


class TrainResponse(PluginDTO):
    """Train response: fields match the dict returned by plugin.train()."""

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
    mlflow_run_id: ConfirmedRunId = ""
