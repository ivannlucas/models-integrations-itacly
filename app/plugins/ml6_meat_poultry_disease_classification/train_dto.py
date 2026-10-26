"""Pydantic request/response DTOs for the ml6 poultry disease /train endpoint.

Retrains the three base models of ONE ensemble with the AI team's procedure
(src/training/trainer.py::train_model, config/config.yaml) on a user dataset with the same
layout as the original raw datasets. See manifest.training in inbox/a06/manifest.yaml.
"""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.application.dto.train_dto import MlflowRunId


class TrainRequest(BaseModel):
    """Train request: the raw dataset of one modality, split 70/15/15 with seed 42."""

    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(
        ...,
        description=(
            "Ruta local, s3:// o .zip con una subcarpeta por clase, igual que data/<tarea>/raw "
            "del código original: thermal → healthy/ y sick/; rgb → pcrhealthy/, pcrcocci/, "
            "pcrncd/ y pcrsalmo/ (también se aceptan healthy/, cocci/, ncd/, salmo/). Se divide "
            "70/15/15 con seed 42 igual que src/utils/io.py::ensure_data_splits."
        ),
    )
    mlflow_run_id: MlflowRunId = Field(
        ...,
        description=(
            "Run de MLflow donde se persiste el ensemble reentrenado. Obligatorio: el artefacto "
            "base nunca se sobrescribe en el sitio."
        ),
    )
    task: Literal["thermal", "rgb"] = Field(..., description="Ensemble a reentrenar")


class TrainResponse(BaseModel):
    """Train response: test-split metrics of each retrained base model and of the ensemble."""

    model_config = ConfigDict(protected_namespaces=())
    detail: str
    task: str
    accuracy: float = Field(..., description="Accuracy del ensemble en el split de test")
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    per_class: dict[str, dict[str, float]] = Field(
        ..., description="Precision/recall/f1 por clase del ensemble en test (memoria Tabla 6)",
    )
    models: dict[str, dict[str, Any]] = Field(
        ...,
        description="Por modelo base: métricas de test, mejor accuracy de validación y épocas",
    )
    n_train: int
    n_val: int
    n_test: int
    training_time_s: float
    mlflow_run_id: str = ""
    upload_warning: str | None = Field(
        default=None, description="Siempre None: un fallo al guardar en MLflow responde 502",
    )
