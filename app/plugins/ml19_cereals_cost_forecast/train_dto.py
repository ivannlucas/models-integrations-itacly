"""Pydantic DTOs for the ml19 /train endpoint."""
from pydantic import BaseModel, Field

from app.application.dto.plugin_dto import PluginDTO, PromotingUploadWarning, PublishedBaseVersion


class TrainRequest(PluginDTO):
    """Retrain with the AI team's official pipeline on an updated dataset_v7_fe.csv."""

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


class SelectedRegMetrics(PluginDTO):
    """Walk-forward metrics (out-of-sample, from TRAIN_END) of the selected regressor."""

    model_name: str
    corr: float
    dir_acc: float
    sharpe: float
    mae: float
    rmse: float
    r2: float


class SelectedClfMetrics(PluginDTO):
    """Walk-forward metrics (out-of-sample, from TRAIN_END) of the selected classifier."""

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


class TrainResponse(PluginDTO):
    """Walk-forward metrics of the selected models, comparable with memoria Tables 38-39."""

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
    upload_warning: PromotingUploadWarning = None
    base_version: PublishedBaseVersion = None
