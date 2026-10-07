"""Pydantic DTOs for the fungal leaf-disease CNN ``/train`` endpoint.

``mlflow_run_id`` has no default — it is required on purpose. Every retrain must be
persisted to a specific MLflow run; the retrained LeafCNN is never written to the local/S3
base artifact (``leafcnn_best.pth``), only uploaded to that MLflow run's ``model`` artifact
path, exactly like every other plugin in this repo with real training support
(ml10_dairy_disease_vector_detection, ml8_cereals_img_anomaly_detector, ml30_meat_traceability_detection).
"""
from pydantic import BaseModel, ConfigDict, Field

from app.application.dto.train_dto import MlflowRunId


class TrainRequest(BaseModel):
    """Train request: a ZIP with one subfolder per class, each full of leaf images."""

    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(
        ...,
        description=(
            "Ruta (local o s3://) a un ZIP con una subcarpeta por clase "
            "(black_rot/, downy_mildew/, healthy/, powdery_mildew/, trunk_disease/), cada una "
            "con imágenes JPG/PNG/BMP — misma convención de carpetas-por-clase que "
            "data/processed/Images_256 en el código de entrenamiento original."
        ),
    )
    mlflow_run_id: MlflowRunId


class TrainResponse(BaseModel):
    """Train response: metrics computed on the held-out validation split."""

    model_config = ConfigDict(protected_namespaces=())
    detail: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    n_train: int
    n_val: int
    classes: list[str]
    epochs_run: int
    upload_warning: str | None = Field(
        default=None,
        description="Informativo si el entrenamiento terminó pero falló la subida a MLflow",
    )
    mlflow_run_id: str = Field(default="")
