"""Pydantic DTOs for the ml3 wine disease/pest /train endpoint."""
from pydantic import Field

from app.application.dto.plugin_dto import PluginDTO, TrainRequestBase


class TrainRequest(TrainRequestBase):
    """Retrain request: labeled raw CSV/parquet with the full training contract columns."""

    data_path: str = Field(
        ...,
        description=(
            "Ruta (local o s3://) a un CSV o parquet etiquetado con las columnas del contrato "
            "de entrenamiento (inbox/a03/manifest.yaml -> training.required_columns): las 8 "
            "variables de sensor, Fecha, ID_Serie, Clase_Entrenamiento y Grado_Infeccion. El "
            "plugin aplica apply_feature_engineering sobre la serie completa, hace el split "
            "70/15/15 por ID_Serie (seed 42), reentrena el ensemble completo (LSTM + CNN + "
            "BiGRU) desde cero y devuelve las métricas de test hold-out."
        ),
    )


class TrainResponse(PluginDTO):
    """Retrain response: hold-out metrics of the full Deep Ensemble pipeline."""

    detail: str
    n_windows_train: int
    n_windows_val: int
    n_windows_test: int
    epochs_executed: int = Field(
        ..., description="Épocas efectivas por arquitectura (EarlyStopping)"
    )
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    f1_weighted: float
    mae: float
    mse: float
    r2: float
    upload_warning: str | None = Field(
        default=None,
        description=(
            "Aviso si el reentrenamiento terminó pero el modelo no se ha podido guardar en MLflow"
        ),
    )
