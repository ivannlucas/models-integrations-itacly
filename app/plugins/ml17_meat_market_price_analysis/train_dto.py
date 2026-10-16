"""Train DTOs for ml17 — Ridge pork price forecast retraining."""
from pydantic import BaseModel, ConfigDict, Field

from app.application.dto.base_version_dto import VersionInfo


class TrainRequest(BaseModel):
    """Retrain request: CSV with the 8 official_v1_4 columns; MLflow run optional."""

    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(
        ...,
        description=(
            "Ruta al CSV mensual con columnas date, target_price_pigmeat_class_e_es, "
            "eurostat_pigmeat_slaughter_tonnes_es, eurostat_pigmeat_slaughter_tonnes_eu, "
            "cereal_feed_barley_price_monthly, cereal_feed_maize_price_monthly, "
            "mapa_porcino_otras_razas_price_monthly, month_sin, month_cos — ordenado "
            "cronológicamente, mínimo 2 filas (supervisión one-step-ahead t -> t+1)."
        ),
    )
    mlflow_run_id: str = Field(
        default="",
        description=(
            "Run de MLflow opcional: si se indica, el modelo reentrenado también se guarda en él "
            "(trazabilidad). Con o sin run, el modelo reentrenado pasa a ser el modelo base."
        ),
    )


class TrainResponse(BaseModel):
    """Retrain response: in-sample one-step metrics of the refit Ridge pipeline."""

    model_config = ConfigDict(protected_namespaces=())
    detail: str
    mae: float = Field(..., description="MAE one-step in-sample tras el reentrenamiento (€/100kg)")
    rmse: float = Field(..., description="RMSE one-step in-sample tras el reentrenamiento (€/100kg)")
    mase: float | None = Field(
        default=None, description="MASE respecto a la escala de la propia serie de entrenamiento"
    )
    r2_train: float | None = Field(default=None, description="R² in-sample tras el reentrenamiento")
    directional_accuracy: float | None = Field(
        default=None,
        description="Exactitud direccional in-sample sobre llamadas no planas (puede ser null si no hay ninguna)",
    )
    n_samples: int = Field(..., description="Número de pares (t, t+1) usados para el ajuste")
    upload_warning: str | None = Field(
        default=None, description="Aviso si el modelo reentrenado no se ha podido guardar en MLflow"
    )
    base_version: VersionInfo | None = Field(
        default=None, description="Versión del modelo base publicada por este reentreno"
    )
