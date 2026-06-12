from pydantic import BaseModel, ConfigDict, Field


class TrainRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(
        ...,
        description=(
            "Ruta a un fichero ZIP dentro del contenedor con estructura "
            "{cereal}/train/{categoria}/*.jpg"
        ),
    )


class TrainResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
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
