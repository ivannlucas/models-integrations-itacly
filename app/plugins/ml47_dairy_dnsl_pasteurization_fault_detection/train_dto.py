from pydantic import BaseModel, ConfigDict, Field


class TrainRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(..., description="CSV with sensor features + target columns")
    mlflow_run_id: str


class TrainResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    detail: str
    exact_match: float
    accuracy: float
    f1_macro: float
    n_train: int
    n_test: int
    training_time_s: float
    upload_warning: str | None = None
