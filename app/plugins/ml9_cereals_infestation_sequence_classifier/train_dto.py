"""Pydantic DTOs for the ml9 (cereal infestation) /train endpoint."""
from pydantic import BaseModel, ConfigDict, Field


class TrainRequest(BaseModel):
    """Fine-tune request: CSV of labelled hourly telemetry (manifest.training.required_columns)."""

    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(
        ...,
        description=(
            "Ruta (local o s3://) a un CSV con sample_id, timestamp, co2_ppm, temp_c, "
            "ambient_rh_pct, humidity_grain_pct y target (0=sano, 1=insectos, 2=moho_critico). Se "
            "recomienda incluir target_global para estratificar la partición por serie igual que el "
            "entrenamiento original. Requisitos mínimos: al menos 3 sample_id por clase y al menos "
            "48 observaciones horarias por sample_id."
        ),
    )
    mlflow_run_id: str = Field(default="", description="MLflow run ID donde registrar métricas y artefactos")


class TrainResponse(BaseModel):
    """Fine-tune response: hold-out metrics of the retrained model on the caller's own CSV."""

    model_config = ConfigDict(protected_namespaces=())
    detail: str
    n_series_train: int
    n_series_validation: int
    n_series_test: int
    n_windows_train: int
    n_windows_validation: int
    n_windows_test: int
    epochs_run: int = Field(..., description="Época en la que se alcanzó el mejor f1_macro de validación")
    # manifest.training.metrics_returned — medidas sobre el hold-out del CSV aportado
    accuracy: float
    balanced_accuracy: float
    f1_macro: float
    precision_macro: float
    recall_macro: float
    log_loss: float
    validation_f1_macro: float = Field(..., description="Mejor f1_macro de validación (criterio de early stopping)")
    baseline_f1_macro: float = Field(
        ...,
        description="f1_macro del modelo servido sobre el mismo hold-out, para comparar antes/después",
    )
    artifact_path: str = Field(..., description="Ruta local del checkpoint reentrenado (nunca sobrescribe el fijo)")
    upload_warning: str | None = Field(
        default=None,
        description="Informativo si el fine-tuning terminó pero falló la subida a MLflow",
    )
