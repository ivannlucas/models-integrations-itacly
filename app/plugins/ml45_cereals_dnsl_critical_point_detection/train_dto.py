from pydantic import BaseModel, ConfigDict, Field


class TrainRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(
        ...,
        description=(
            "Path to CSV with timestamp, cycle_id, the 11 sensor columns, and fault_name "
            "(ground truth, binarized internally via normal_tokens)."
        ),
    )
    mlflow_run_id: str = ""


class TrainResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    detail: str
    accuracy: float
    f1: float
    auc: float
    n_windows: int
    n_epochs: int
