"""Pydantic DTOs for the ml18 /train endpoint."""
# pylint: disable=too-few-public-methods  # Pydantic DTOs carry no behaviour
from pydantic import BaseModel, ConfigDict, Field

from app.application.dto.train_dto import MlflowRunId


class TrainRequest(BaseModel):
    """Retrain the GRU from scratch with the AI team's procedure on a modelling CSV."""

    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(
        ...,
        description=(
            "Ruta local o s3:// a un CSV con separador ';' en el formato de "
            "dataset_modelado_desde_2008.csv: Fecha, CCAA, Producto, Poblacion, RentaHogar, "
            "CONSUMO X CAPITA, PENETRACION (%) y PRECIO MEDIO KG, una fila por mes y combinación "
            "CCAA-Producto. Los lags espaciales y el precio propio se recalculan. Split temporal "
            "70/15/15 por fecha; cada combinación necesita más de 12 meses."
        ),
    )
    mlflow_run_id: MlflowRunId = Field(
        ...,
        description="Run de MLflow donde se persiste el modelo reentrenado. Obligatorio: el "
        "artefacto base nunca se sobrescribe.",
    )


class TrainResponse(BaseModel):
    """Train/val/test metrics in euros (same as models/artifacts/.../metrics.json)."""

    model_config = ConfigDict(protected_namespaces=())
    detail: str
    train_mae: float
    train_rmse: float
    train_mape_pct: float
    train_r2: float
    val_mae: float
    val_rmse: float
    val_mape_pct: float
    val_r2: float
    test_mae: float
    test_rmse: float
    test_mape_pct: float
    test_r2: float
    n_train: int
    n_val: int
    n_test: int
    epochs_run: int
    best_epoch: int
    training_time_s: float
    mlflow_run_id: str
    upload_warning: str | None = Field(
        default=None,
        description="Compatibilidad con la plataforma: siempre None (un fallo de subida es un 502)",
    )
