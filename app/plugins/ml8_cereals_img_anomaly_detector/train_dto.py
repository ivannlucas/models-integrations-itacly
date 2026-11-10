"""Pydantic DTOs for the ml8-cereals-img-anomaly-detector /train endpoint."""
from pydantic import Field

from app.application.dto.plugin_dto import ConfirmedRunId, PluginDTO, TrainRequestBase


class TrainRequest(TrainRequestBase):
    """Training request: ZIP with {cereal}/train/{categoria}/ images."""

    data_path: str = Field(
        ...,
        description=(
            "Ruta a un fichero ZIP dentro del contenedor con estructura "
            "{cereal}/train/{categoria}/*.jpg"
        ),
    )


class TrainResponse(PluginDTO):
    """Training result: sample counts, phase timings and best validation accuracies."""

    detail: str
    train_samples: int
    val_samples: int
    fase1_epochs: int
    fase2_epochs: int
    fase1_time_min: float
    fase2_time_min: float
    best_val_acc_cat: float
    best_val_acc_cer: float
    upload_warning: str | None = Field(
        default=None,
        description="Informativo si los artefactos se guardaron en local pero falló el upload a S3",
    )
    mlflow_run_id: ConfirmedRunId = ""
