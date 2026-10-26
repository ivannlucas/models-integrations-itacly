"""Train DTOs for ml1 — RandomForest tabular experts only (see constants.py / manifest.yaml
training.scope_note: the CNN Deep Ensemble + YOLO are not retrainable via this endpoint)."""
from pydantic import BaseModel, ConfigDict, Field

from app.application.dto.train_dto import MlflowRunId


class TrainRequest(BaseModel):
    """Retrain the 2 RandomForest tabular experts on a client-supplied sensor CSV."""

    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(
        ...,
        description=(
            "Ruta local o s3:// a un CSV con columnas: Mes, Temp_Max_Media_5d_C, "
            "Temp_Min_Media_5d_C, Hum_Rel_Max_5d_Pct, Hum_Rel_Min_5d_Pct, Hum_Suelo_Pct, "
            "Lluvia_Acum_5d_mm, Horas_Humedad_Foliar, GDD_Acumulado, Etiqueta, "
            "Indice_Riesgo_Cientifico."
        ),
    )
    mlflow_run_id: MlflowRunId = Field(
        ..., description="Run de MLflow donde se persiste el par RF reentrenado — obligatorio."
    )


class TrainResponse(BaseModel):
    """Metrics of the retrained RandomForest pair, holdout test (same protocol as
    src/training/train_tabular.py: 80/20 split, random_state=42)."""

    model_config = ConfigDict(protected_namespaces=())
    detail: str
    accuracy: float
    f1_macro: float
    f1_weighted: float
    mae_riesgo: float
    n_train: int
    n_test: int
    upload_warning: str | None = None
