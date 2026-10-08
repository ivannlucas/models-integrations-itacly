"""Pydantic DTO for ml10-dairy-disease-vector-detection /train endpoint.

The response reuses app.application.dto.train_dto.TrainResponse directly — train() already
builds and returns that exact class (detail + metrics dict + mlflow_run_id + upload_warning),
so there is nothing plugin-specific to add there. The request needed its own type because the
generic default TrainRequest's `data_path` has no default and this plugin's positional contract
differs slightly; `mlflow_run_id` is required here on purpose — every retrain must be persisted
to a specific MLflow run (never silently discarded, never overwrites the fixed S3 artifact),
exactly like every other plugin with real training support.
"""
from pydantic import BaseModel, ConfigDict, Field

from app.application.dto.train_dto import MlflowRunId


class TrainRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(
        ...,
        description=(
            "Ruta a un .zip con imagenes de entrenamiento. Estructuras aceptadas: plana "
            "({fly|mos|tick}/*.jpg, auto-split 70/15/15) o pre-dividida "
            "({train|val}/{fly|mos|tick}/*.jpg)."
        ),
    )
    mlflow_run_id: MlflowRunId
