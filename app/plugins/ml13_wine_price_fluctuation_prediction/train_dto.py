"""Pydantic request/response DTOs for the ml13 /train endpoint."""
from pydantic import BaseModel, ConfigDict, Field

from app.application.dto.train_dto import MlflowRunId


class TrainRequest(BaseModel):
    """Retrain request: raw weekly price CSV (the binary target is derived from the price)."""

    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(
        ...,
        description=(
            "Ruta local o s3:// a un CSV semanal con 'campaign'+'week' (o 'bulletin') y "
            "'price_red' "
            "(o otra columna de precio candidata). El target se calcula internamente: "
            "1 si la media del precio en t+1..t+4 supera el precio_t en más de un 2,5%. Se repite "
            "el procedimiento original: walk-forward CV (5 folds, gap 4) LogReg vs XGBoost, "
            "selección por Smart Score, ajuste final sobre todo salvo las 24 últimas semanas y "
            "evaluación hold-out sobre esas 24."
        ),
    )
    mlflow_run_id: MlflowRunId = Field(
        ...,
        description="Run de MLflow donde se persiste el modelo reentrenado. Obligatorio: el "
        "artefacto base nunca se sobrescribe.",
    )


class TrainResponse(BaseModel):
    """Retrain response: hold-out metrics (last 24 weeks) + walk-forward CV summary."""

    model_config = ConfigDict(protected_namespaces=())
    detail: str
    best_model_type: str
    n_trainval_rows: int
    n_test_rows: int
    test_period: str
    auc: float
    accuracy: float
    f1: float
    precision: float
    recall: float
    cv_logreg_auc_mean: float
    cv_logreg_auc_std: float
    cv_logreg_f1_mean: float
    cv_xgboost_auc_mean: float
    cv_xgboost_auc_std: float
    cv_xgboost_f1_mean: float
    smart_score_logreg: float
    smart_score_xgboost: float
    upload_warning: str | None = Field(
        default=None,
        description="Compatibilidad con la plataforma: siempre None (un fallo de subida es un 502)",
    )
