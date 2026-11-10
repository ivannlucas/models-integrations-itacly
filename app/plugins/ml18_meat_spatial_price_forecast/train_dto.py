"""Pydantic DTOs for the ml18 /train endpoint."""
from pydantic import Field

from app.application.dto.plugin_dto import PluginDTO, PromotingTrainRunId, PublishedBaseVersion


class TrainRequest(PluginDTO):
    """Retrain the GRU from scratch with the AI team's procedure on a modelling CSV."""

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
    mlflow_run_id: PromotingTrainRunId = ""


class TrainResponse(PluginDTO):
    """Train/val/test metrics in euros (same as models/artifacts/.../metrics.json)."""

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
    mlflow_run_id: str = ""
    upload_warning: str | None = Field(
        default=None,
        description=(
            "Aviso si falló la subida a MLflow; el modelo base se publica igualmente "
            "(502 solo si falla la publicación del base)"
        ),
    )
    base_version: PublishedBaseVersion = None
