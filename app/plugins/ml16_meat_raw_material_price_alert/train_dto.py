"""Pydantic request/response DTOs for the ml16 /train endpoint."""
from pydantic import BaseModel, ConfigDict, Field


class TrainRequest(BaseModel):
    """Retrain request: labeled CSV shaped like dataset_clasificacion_base.csv."""

    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(
        ...,
        description=(
            "Ruta a un CSV con fecha, month, indice_animales, indice_insumos, precip_total, "
            "precip_max, wet_days, wash_days, animales_afectados, target_animales, "
            "target_insumos (targets ya calculados — este endpoint no reproduce create_targets() "
            "ni la ETL cruda de MAPA/GEE/RASVE, solo la etapa de modelado). Se reentrenan "
            "XGBoost y LogisticRegression desde cero con los hiperparámetros originales "
            "(walk-forward CV + búsqueda de umbral + bagging bootstrap)."
        ),
    )
    mlflow_run_id: str = ""


class TrainResponse(BaseModel):
    """Retrain response: hold-out test metrics per target (walk-forward CV + bagging)."""

    model_config = ConfigDict(protected_namespaces=())
    detail: str
    n_train_rows: int
    n_test_rows: int
    target_animales_threshold: float
    target_animales_accuracy: float
    target_animales_precision: float
    target_animales_recall: float
    target_animales_f1: float
    target_animales_auc: float
    target_insumos_threshold: float
    target_insumos_accuracy: float
    target_insumos_precision: float
    target_insumos_recall: float
    target_insumos_f1: float
    target_insumos_auc: float
    upload_warning: str | None = None
