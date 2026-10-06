"""Pydantic DTO for ml5-meat-cow-behaviour's ``/train`` endpoint.

Only the SlowFast behaviour CLASSIFIER supports retraining through this endpoint — the
Faster R-CNN DETECTOR remains out of scope (bbox-annotation-only ZIP contract mismatch,
documented in inbox/a05/manifest.yaml training.reason, unrelated to the 4 training-code
bugs fixed this cycle). The response reuses app.application.dto.train_dto.TrainResponse
directly — train() already builds and returns that exact class (detail + metrics dict +
mlflow_run_id + upload_warning), same pattern as ml10_dairy_disease_vector_detection. ``mlflow_run_id`` is
required on purpose — every retrain must be persisted to a specific MLflow run (never
silently discarded, never overwrites the fixed S3 artifact).
"""
from pydantic import BaseModel, ConfigDict, Field


class TrainRequest(BaseModel):
    """Request body for retraining the SlowFast cow-behaviour classifier."""

    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(
        ...,
        description=(
            "Ruta a un .zip con el dataset de clips anotados para el clasificador SlowFast "
            "(el detector Faster R-CNN permanece fuera de alcance de /train). Estructura "
            "esperada — la misma que ya consume scripts/data/format.py del entregable, no "
            "inventada — en la raíz del ZIP: "
            "'raw_frames/<clip_name>/<frame>.jpg' + "
            "'annotations/<clip_name>/annotations/instances_default.json' "
            "(export COCO de CVAT por clip, con attributes.track_id y attributes.behavior "
            "por frame). Admite s3://<bucket>/<key>.zip."
        ),
    )
    mlflow_run_id: str = Field(
        ...,
        description=(
            "MLflow run ID donde se persiste el clasificador reentrenado. Obligatorio: "
            "el reentrenamiento nunca se descarta silenciosamente ni sobreescribe el "
            "artefacto base fijo servido por la plataforma."
        ),
    )
