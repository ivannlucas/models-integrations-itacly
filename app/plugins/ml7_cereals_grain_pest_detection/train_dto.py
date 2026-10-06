"""Train DTOs for ml7 grain pest-detection YOLO fine-tuning."""
from pydantic import BaseModel, ConfigDict, Field


class TrainRequest(BaseModel):
    """Fine-tune the YOLO detector on a new annotated image dataset.

    ``data_path`` must point to a YOLO-format dataset root — a directory containing
    ``images/{train,val,test}`` + ``labels/{train,val,test}`` and a ``dataset.yaml``
    describing them (same layout ``scripts/data_processing.py`` produces in the
    delivered code from raw Pascal VOC XML annotations). There is no CSV-based training
    for this model — it is an object detector, not a tabular regressor/classifier.
    """

    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(
        ...,
        description=(
            "Path to a YOLO-format dataset root (images/{train,val,test} + "
            "labels/{train,val,test} + dataset.yaml), or directly to a dataset.yaml file."
        ),
    )
    mlflow_run_id: str = Field(
        ..., description="MLflow run ID to persist the fine-tuned checkpoint to (required)."
    )


class TrainResponse(BaseModel):
    """Fine-tuning result — detection metrics computed on the dataset's val split."""

    model_config = ConfigDict(protected_namespaces=())
    detail: str
    map50: float = Field(..., description="mAP@0.5 on the val split after fine-tuning")
    map50_95: float = Field(..., description="mAP@0.5:0.95 on the val split after fine-tuning")
    precision: float = Field(..., description="Mean precision on the val split")
    recall: float = Field(..., description="Mean recall on the val split")
    n_images: int = Field(..., description="Number of training images used")
    upload_warning: str | None = Field(
        default=None, description="Aviso si el modelo reentrenado no se ha guardado en MLflow"
    )
