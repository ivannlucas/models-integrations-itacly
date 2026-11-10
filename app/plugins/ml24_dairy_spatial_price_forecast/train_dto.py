"""Pydantic DTOs for the ml24 /train endpoint."""
from pydantic import BaseModel, Field

from app.application.dto.plugin_dto import PluginDTO, PromotingTrainRunId, PublishedBaseVersion


class TrainRequest(PluginDTO):
    """Retrain the three horizons from scratch with the AI team's procedure on a series CSV."""

    data_path: str = Field(
        ...,
        description=(
            "Ruta local o s3:// a un CSV en el formato largo de series_consumo_lacteos_ccaa.csv: "
            "anio, "
            "mes, ccaa, producto, metrica, valor, es_reconstruido, con TOTAL LECHE LIQUIDA y los 4 "
            "productos de contexto. Los paneles, rezagos y señales espaciales se recalculan; la "
            "matriz "
            "de fronteras es la del modelo base. La serie (sus 5 productos) pasa a ser la del "
            "modelo "
            "base nuevo, con la que se predice."
        ),
    )
    mlflow_run_id: PromotingTrainRunId = ""
    exog_path: str = Field(
        default="",
        description=(
            "Opcional: CSV con el formato de input_costs_monthly.csv (exógenas mensuales del "
            "mercado "
            "lácteo UE, period_dt + columnas exog__*). Si se indica, se entrena con él y pasa a "
            "ser el "
            "del modelo base nuevo; si no, se usan y se heredan las vigentes (upload_warning lo "
            "avisa)."
        ),
    )
    validation_year: int | None = Field(
        default=None, ge=2005, le=2100,
        description=(
            "Año de validación. Por defecto test_year - 1. config.yaml del equipo: 2019. Train = "
            "fechas objetivo anteriores."
        ),
    )
    test_year: int | None = Field(
        default=None, ge=2006, le=2100,
        description=(
            "Año de test = validation_year + 1. Por defecto, el último año con los 12 meses del "
            "precio "
            "ancla en la serie (config.yaml del equipo: 2020). Los paneles se cortan en este año."
        ),
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


class TrainResponse(PluginDTO):
    """Per-horizon metrics of the retraining, comparable with the memoria's Tables 8-10."""

    detail: str
    h1: HorizonMetrics
    h3: HorizonMetrics
    h6: HorizonMetrics
    validation_year: int
    test_year: int
    task_type: str
    training_time_s: float
    mlflow_run_id: str = ""
    exog_renewed: bool = Field(default=False, description="True si /train recibió exog_path")
    upload_warning: str | None = Field(
        default=None,
        description=(
            "Avisos: subida a MLflow fallida (el modelo base se publica igualmente; solo un fallo "
            "al "
            "publicar la base es un 502) o exógenas no renovadas"
        ),
    )
    base_version: PublishedBaseVersion = None
