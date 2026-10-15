"""Pydantic DTOs for ml21 (ESP-CEREAL) /train endpoint."""
from pydantic import BaseModel, ConfigDict, Field

from app.application.dto.base_version_dto import VersionInfo


class TrainRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(
        ..., description="Path to CSV with raw panel rows + target columns"
    )
    mlflow_run_id: str = Field(
        default="",
        description=(
            "Run de MLflow opcional: si se indica, el modelo reentrenado también se guarda en él "
            "(trazabilidad). Con o sin run, el modelo reentrenado pasa a ser el modelo base y "
            "este CSV su panel (dataset_entrenamiento_final.csv)."
        ),
    )


class TrainResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    detail: str
    mae_h1: float | None = None
    mae_h2: float | None = None
    mae_h3: float | None = None
    pearson_h1: float | None = None
    pearson_h2: float | None = None
    pearson_h3: float | None = None
    da_h1: float | None = None
    da_h2: float | None = None
    da_h3: float | None = None
    auc_h1: float | None = None
    auc_h2: float | None = None
    auc_h3: float | None = None
    n_train: int | None = None
    n_test: int | None = None
    upload_warning: str | None = None
    base_version: VersionInfo | None = None
