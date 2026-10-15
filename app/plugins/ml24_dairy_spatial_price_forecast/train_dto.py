"""Pydantic DTOs for the ml24 /train endpoint."""
# pylint: disable=line-too-long  # repo limit is 120 (.flake8), enforced by flake8
# pylint: disable=too-few-public-methods  # Pydantic DTOs carry no behaviour
from pydantic import BaseModel, ConfigDict, Field

from app.application.dto.train_dto import MlflowRunId
from app.plugins.ml24_dairy_spatial_price_forecast.constants import TRAIN_TEST_YEAR, TRAIN_VALIDATION_YEAR


class TrainRequest(BaseModel):
    """Retrain the three horizons from scratch with the AI team's procedure on a series CSV."""

    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(
        ...,
        description=(
            "Ruta local o s3:// a un CSV en el formato largo de series_consumo_lacteos_ccaa.csv: anio, "
            "mes, ccaa, producto, metrica, valor, es_reconstruido, con TOTAL LECHE LIQUIDA y los 4 "
            "productos de contexto. Los paneles, rezagos y señales espaciales se recalculan; la matriz "
            "de fronteras y las exógenas mensuales son las del artefacto base."
        ),
    )
    mlflow_run_id: MlflowRunId = Field(
        ...,
        description="Run de MLflow donde se persiste el modelo reentrenado. Obligatorio: el "
        "artefacto base nunca se sobrescribe.",
    )
    validation_year: int = Field(
        default=TRAIN_VALIDATION_YEAR, ge=2005, le=2100,
        description="Año de validación (config.yaml: 2019). Train = fechas objetivo anteriores.",
    )
    test_year: int = Field(
        default=TRAIN_TEST_YEAR, ge=2006, le=2100,
        description="Año de test = validation_year + 1 (config.yaml: 2020). Los paneles se cortan en este año.",
    )


class HorizonSplitMetrics(BaseModel):
    """MAE / sMAPE of the model and of the calibrated baseline on one split."""

    mae: float
    smape: float
    mae_baseline: float
    smape_baseline: float
    accuracy_pct: float
    n: int


class HorizonMetrics(BaseModel):
    """Train, validation and test metrics of one horizon (models/metrics/.../metrics.csv)."""

    train: HorizonSplitMetrics
    validation: HorizonSplitMetrics
    test: HorizonSplitMetrics
    baseline_w_persist_1: float
    baseline_w_persist_12: float
    ml_alpha: float
    n_features: int


class TrainResponse(BaseModel):
    """Per-horizon metrics of the retraining, comparable with the memoria's Tables 8-10."""

    model_config = ConfigDict(protected_namespaces=())
    detail: str
    h1: HorizonMetrics
    h3: HorizonMetrics
    h6: HorizonMetrics
    validation_year: int
    test_year: int
    task_type: str
    training_time_s: float
    mlflow_run_id: str
    upload_warning: str | None = Field(
        default=None,
        description="Compatibilidad con la plataforma: siempre None (un fallo de subida es un 502)",
    )
