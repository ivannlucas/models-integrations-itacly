"""Pydantic DTOs for the ml46 (DNSL fouling/clog) /train endpoint."""
from pydantic import BaseModel, ConfigDict, Field


class TrainRequest(BaseModel):
    """Fine-tune request: CSV with timestamp, asset_id, Rf_m2K_W plus the raw contract columns."""

    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(
        ...,
        description=(
            "CSV con, como mínimo, timestamp, asset_id y Rf_m2K_W (severidad objetivo), más las "
            "columnas del contrato de entrada. Se recomienda aportar el resto de columnas crudas "
            "(flow_kg_s, dP_kPa, temperaturas, composición nominal, phase, maintenance_active...) "
            "para que la ingeniería de características no dependa de valores por defecto."
        ),
    )
    mlflow_run_id: str = ""


class TrainResponse(BaseModel):
    """Fine-tune response: window-level metrics computed on the caller's own CSV."""

    model_config = ConfigDict(protected_namespaces=())
    detail: str
    n_windows: int
    epochs: int
    severity_rmse: float
    severity_mae: float
    stage_accuracy: float
    stage_macro_f1: float
    watch_foul_auc: float | None = None
    watch_foul_ap: float | None = None
    clog_h_auc: float | None = None
    clog_h_ap: float | None = None
    tte_foul_mae_min: float
    tte_clog_mae_min: float
    ttu_mae_min: float
    upload_warning: str | None = Field(
        default=None,
        description="Informativo si los artefactos se guardaron en local pero falló el upload a S3/MLflow",
    )
