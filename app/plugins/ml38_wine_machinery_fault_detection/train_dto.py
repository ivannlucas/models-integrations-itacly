"""Pydantic DTOs for the ml38 /train endpoint (retraining as in ``python -m scripts.train``)."""
from pydantic import Field

from app.application.dto.plugin_dto import (
    ConfirmedRunId,
    PluginDTO,
    TrainRequestBase,
    UploadWarning,
)
from app.application.dto.train_dto import MlflowRunId


class TrainRequest(TrainRequestBase):
    """Retraining request: a feature table like data/processed/feature_dataset.parquet."""

    data_path: str = Field(
        ...,
        description=(
            "Ruta (local o s3://) a un CSV o parquet con las 24 características (vib_*, ele_*), "
            "family (familia de fallo), severity (entero; -1 si no aplica; 1 = leve, >1 = "
            "significativo) y group_id (ensayo de origen, para la validación cruzada agrupada). "
            "Se necesitan al menos 2 grupos."
        ),
    )
    mlflow_run_id: MlflowRunId = Field(
        ...,
        description=(
            "Run de MLflow donde se persiste el modelo reentrenado. Obligatorio: el artefacto "
            "base nunca se sobrescribe."
        ),
    )


class SeverityCvMetrics(PluginDTO):
    """GroupKFold CV metrics of one family's severity MLP (training_severity_metrics.json)."""

    accuracy_mean: float
    accuracy_std: float
    macro_f1_mean: float
    macro_f1_std: float


class TrainResponse(PluginDTO):
    """Retraining result: the same CV metrics scripts.train writes, plus counts and timing."""

    detail: str
    family_cv_accuracy_mean: float = Field(..., description="Accuracy media por fold (GroupKFold)")
    family_cv_accuracy_std: float
    family_cv_macro_f1_mean: float = Field(..., description="Macro-F1 medio por fold (GroupKFold)")
    family_cv_macro_f1_std: float
    severity_cv: dict[str, SeverityCvMetrics] = Field(
        default_factory=dict, description="Métricas CV del MLP de severidad de cada familia"
    )
    severity_skipped: dict[str, str] = Field(
        default_factory=dict,
        description="Familias sin modelo de severidad y motivo (como en el original)",
    )
    families: list[str] = Field(default_factory=list, description="Clases del nuevo clasificador")
    n_samples: int
    n_groups: int
    training_time_s: float
    mlflow_run_id: ConfirmedRunId = ""
    upload_warning: UploadWarning = None
