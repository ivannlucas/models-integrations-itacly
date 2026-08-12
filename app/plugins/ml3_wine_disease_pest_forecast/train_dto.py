"""Pydantic DTOs for the ml3 wine disease/pest /train endpoint."""
from pydantic import BaseModel, ConfigDict, Field


class TrainRequest(BaseModel):
    """Retrain request: labeled raw CSV/parquet with the full training contract columns."""

    model_config = ConfigDict(protected_namespaces=())
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
    mlflow_run_id: str = ""


class TrainResponse(BaseModel):
    """Retrain response: hold-out metrics of the full Deep Ensemble pipeline."""

    model_config = ConfigDict(protected_namespaces=())
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
            "Informativo si los artefactos se guardaron en local pero falló el upload a MLflow"
        ),
    )
