"""Pydantic request/response DTOs for the thermal-mastitis /train endpoint.

Fine-tunes a fresh clone of the served EfficientNet-B0 on user-supplied labeled thermal
images, following the original training recipe (Adam + Focal Loss + ReduceLROnPlateau +
early stopping) from configs/baseline_efficientnet.yaml (see manifest.training in
inbox/a04/manifest.yaml).
"""
from pydantic import BaseModel, ConfigDict, Field


class TrainRequest(BaseModel):
    """Train request: a dataset with the same layout as the original TIDS dataset."""

    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(
        ...,
        description=(
            "Ruta local, s3:// o .zip con 'ID_Labels.csv' (columnas 'ID','label'; "
            "label=0 Healthy / 1 SCM) y las imágenes en "
            "TIDS_cropped/healthy/<ID>.jpg o TIDS_cropped/SCM/<ID>.jpg, igual que el "
            "dataset TIDS original (src/data/dataset.py)."
        ),
    )
    mlflow_run_id: str = Field(
        ...,
        description=(
            "Run de MLflow al que se persiste el reentrenamiento (obligatorio: un retrain "
            "siempre se guarda contra un run concreto, nunca sobre el artefacto base fijo)."
        ),
    )


class TrainResponse(BaseModel):
    """Train response: metrics on the held-out validation split of the user's data."""

    model_config = ConfigDict(protected_namespaces=())
    detail: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float | None = Field(
        default=None, description="None si la validación es de una sola clase"
    )
    n_train: int
    n_val: int
    training_time_s: float
    mlflow_run_id: str = Field(
        default="",
        description="Run de MLflow al que se subió el checkpoint reentrenado (model_state_dict).",
    )
    upload_warning: str | None = Field(
        default=None,
        description="Informativo si el modelo se entrenó pero falló la subida a MLflow.",
    )
