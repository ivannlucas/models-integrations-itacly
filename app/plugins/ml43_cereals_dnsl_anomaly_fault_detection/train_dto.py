from pydantic import BaseModel, ConfigDict, Field


class TrainRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(..., description="CSV con las 13 columnas de sensor + fault_name (timestamp obligatorio, cycle_id opcional)")
    mlflow_run_id: str


class TrainResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    detail: str
    accuracy: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    fallo_f1: float
    fallo_precision: float
    fallo_recall: float
    n_train: int
    n_test: int
    n_windows_total: int
    upload_warning: str | None = None
