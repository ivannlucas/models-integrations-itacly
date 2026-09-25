from pydantic import BaseModel, ConfigDict, Field


class TrainRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(
        ...,
        description=(
            "Path to a ZIP file with structure {snr}_{machine}/{machine_id}/{normal,abnormal}/"
            "*.wav (same layout the original raw dataset uses). 'normal' is required per "
            "combination fine-tuned; 'abnormal' is optional — if present, auc/fnr/fpr/recall/"
            "threshold are also computed for that combination, otherwise only training-loss "
            "metrics are returned. The checkpoint is then saved with threshold=null; "
            "prediction requires an explicit threshold override or training with both classes."
        ),
    )
    mlflow_run_id: str = ""


class CombinationTrainMetrics(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    machine: str
    machine_id: str
    snr: str
    best_val_loss: float
    n_train: int
    n_val: int
    n_abnormal_samples: int | None = None
    auc_mse: float | None = None
    auc_maha: float | None = None
    fnr: float | None = None
    fpr: float | None = None
    recall: float | None = None
    threshold: float | None = None


class TrainResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    detail: str
    per_combination: list[CombinationTrainMetrics]
    upload_warning: str | None = Field(
        default=None,
        description="Informational if local save succeeded but the MLflow upload failed",
    )
