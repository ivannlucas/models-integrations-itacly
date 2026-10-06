"""Train DTOs for ml23 — GRU dairy price forecast refit."""
from pydantic import BaseModel, ConfigDict, Field


class TrainRequest(BaseModel):
    """Retrain request: CSV with the same contract as data/processed/
    dataset_forecast_ready.csv (fecha, producto, canal, target_precio_medio + the
    exogenous feature columns) — the AI team's own training input, not an invented
    format."""

    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(
        ...,
        description=(
            "Ruta a un CSV con columnas fecha, producto, canal, target_precio_medio y las "
            "columnas exógenas (lags, medias móviles, HICP/IPC/MAPA) — mismo contrato que "
            "data/processed/dataset_forecast_ready.csv del equipo de IA."
        ),
    )
    mlflow_run_id: str = Field(
        ...,
        description=(
            "Run de MLflow donde se persiste el GRU reentrenado. Obligatorio: el artefacto "
            "fijo servido (gru_model.pt) nunca se sobrescribe con un reentrenamiento de "
            "usuario — el resultado de este train() solo vive en este run de MLflow."
        ),
    )


class TrainResponse(BaseModel):
    """Retrain response: test-set metrics of the refit GRU (best of N_SEEDS by validation
    RMSE — same artifact-selection policy used for the shipped gru_model.pt)."""

    model_config = ConfigDict(protected_namespaces=())
    detail: str
    mae: float = Field(..., description="MAE en el split de test del refit (€/litro)")
    rmse: float = Field(..., description="RMSE en el split de test del refit (€/litro)")
    mape_pct: float = Field(..., description="Error porcentual absoluto medio en test")
    r2: float = Field(..., description="R² en el split de test del refit")
    direction_acc_pct: float = Field(
        ..., description="Acierto en la dirección del cambio de precio frente al valor actual"
    )
    n_train: int = Field(..., description="Nº de secuencias de entrenamiento usadas en el refit")
    n_test: int = Field(..., description="Nº de secuencias de test evaluadas")
    training_time_s: float = Field(..., description="Tiempo de entrenamiento (todas las seeds)")
    upload_warning: str | None = Field(
        default=None, description="Aviso si el modelo reentrenado no se ha podido guardar en MLflow"
    )
