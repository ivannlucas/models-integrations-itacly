"""Pydantic request/response DTOs for the ml40 /train endpoint."""
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.application.dto.train_dto import MlflowRunId


class TrainRequest(BaseModel):
    """Retrain request: labeled raw CSV of one subsystem."""

    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(
        ...,
        description=(
            "Ruta a un CSV crudo etiquetado de UN subsistema (se detecta por columnas): "
            "run_id, time_min, fault_id y las columnas de sensores del contrato "
            "(inbox/a40/manifest.yaml -> training.required_columns). Se reentrena el "
            "RandomForest desde cero con los hiperparámetros originales del equipo de IA."
        ),
    )
    system: Optional[Literal["refrigeracion", "aireado"]] = Field(
        default=None,
        description="Subsistema a reentrenar; el CSV debe corresponder a él. Si se omite se "
                    "detecta por las columnas.",
    )
    mlflow_run_id: MlflowRunId


class TrainResponse(BaseModel):
    """Retrain response: hold-out metrics of the full pipeline (RF + rules + vote)."""

    model_config = ConfigDict(protected_namespaces=())
    detail: str
    system: str
    n_samples: int = Field(..., description="Filas de entrenamiento tras la ingeniería de variables")
    n_runs_train: int
    n_runs_test: int
    accuracy: float = Field(..., description="Accuracy en el 20% de ciclos de hold-out (pipeline completo)")
    f1_macro: float
    precision_macro: float
    recall_macro: float
    upload_warning: str | None = None
