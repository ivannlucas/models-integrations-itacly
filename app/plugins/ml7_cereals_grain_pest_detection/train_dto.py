"""Train DTOs for ml7 grain pest-detection YOLO fine-tuning."""
from pydantic import Field

from app.application.dto.plugin_dto import PluginDTO, TrainRequestBase, UploadWarning
from app.application.dto.train_dto import MlflowRunId


class TrainRequest(TrainRequestBase):
    """Fine-tune the YOLO detector on a new annotated image dataset.

    ``data_path`` points to a YOLO-format dataset — ``images/{train,val[,test]}`` +
    ``labels/{train,val[,test]}`` and a ``dataset.yaml`` with the 5 classes (same layout
    ``scripts/data_processing.py`` produces in the delivered code from raw Pascal VOC XML
    annotations). Usually a ZIP (local or ``s3://``) with the dataset at its root or in one
    top-level folder; the split paths of its yaml are replaced by that layout. A local
    dataset root or ``dataset.yaml`` is used as is. There is no CSV-based training for this
    model — it is an object detector, not a tabular regressor/classifier.
    """

    data_path: str = Field(
        ...,
        description=(
            "ZIP (local or s3://) of a YOLO-format dataset: images/{train,val[,test]} + "
            "labels/{train,val[,test]} + dataset.yaml with the 5 classes. Also accepts a local "
            "dataset root or dataset.yaml."
        ),
    )
    mlflow_run_id: MlflowRunId = Field(
        ..., description="MLflow run ID to persist the fine-tuned checkpoint to (required)."
    )


class TrainResponse(PluginDTO):
    """Fine-tuning result — detection metrics computed on the dataset's val split."""

    detail: str
    map50: float = Field(..., description="mAP@0.5 on the val split after fine-tuning")
    map50_95: float = Field(..., description="mAP@0.5:0.95 on the val split after fine-tuning")
    precision: float = Field(..., description="Mean precision on the val split")
    recall: float = Field(..., description="Mean recall on the val split")
    n_images: int = Field(..., description="Number of training images used")
    upload_warning: UploadWarning = None
