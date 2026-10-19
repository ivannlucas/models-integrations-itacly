"""Pydantic DTOs for the ml19 /train endpoint."""
# pylint: disable=too-few-public-methods  # Pydantic DTOs carry no behaviour
from pydantic import BaseModel, ConfigDict, Field

from app.application.dto.base_version_dto import VersionInfo


class TrainRequest(BaseModel):
    """Retrain with the AI team's official pipeline on an updated dataset_v7_fe.csv."""

    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(
        ...,
        description=(
            "Ruta local o s3:// al dataset_v7_fe.csv que genera el pipeline del equipo de IA "
            "(python -m src.main data_processing): date, target_media, target_ret_next, "
            "target_dir_next y las features de ingeniería. Las filas sin target (frente de "
            "producción) no entran en el entrenamiento pero sí en el dataset de inferencia."
        ),
    )
    mlflow_run_id: str = Field(
        default="",
        description=(
            "Run de MLflow opcional: si se indica, el modelo reentrenado también se guarda en él "
            "(trazabilidad). Con o sin run, el modelo reentrenado pasa a ser el modelo base y "
            "este CSV su dataset de inferencia."
        ),
    )


class SelectedRegMetrics(BaseModel):
    """Walk-forward metrics (out-of-sample, from TRAIN_END) of the selected regressor."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    corr: float
    dir_acc: float
    sharpe: float
    mae: float
    rmse: float
    r2: float


class SelectedClfMetrics(BaseModel):
    """Walk-forward metrics (out-of-sample, from TRAIN_END) of the selected classifier."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    acc: float
    f1: float
    auc: float
    brier: float
    sharpe: float


class HorizonTrainMetrics(BaseModel):
    """Selected models of one horizon (max Sharpe / max AUC, like the original)."""

    horizon: int
    reg: SelectedRegMetrics
    clf: SelectedClfMetrics


class TrainResponse(BaseModel):
    """Walk-forward metrics of the selected models, comparable with memoria Tables 38-39."""

    model_config = ConfigDict(protected_namespaces=())
    detail: str
    horizons: list[HorizonTrainMetrics]
    n_features: int
    lasso_alpha: float
    train_end: str
    n_rows: int
    data_start: str
    data_end: str
    training_time_s: float
    mlflow_run_id: str = ""
    upload_warning: str | None = Field(
        default=None,
        description=(
            "Aviso si la subida a MLflow falló; el modelo base se publica igualmente "
            "(solo un fallo al publicar la base es un 502)"
        ),
    )
    base_version: VersionInfo | None = Field(
        default=None, description="Versión del modelo base publicada por este reentreno"
    )
