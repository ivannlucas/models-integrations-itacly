from pydantic import BaseModel, ConfigDict, Field

from app.application.dto.train_dto import MlflowRunId


class TrainRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(..., description="CSV con columnas de sensor + fault_name (opcional cycle_id/timestamp)")
    mlflow_run_id: MlflowRunId


class TrainResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    detail: str
    accuracy: float
    fallo_auc: float
    fallo_precision: float
    fallo_recall: float
    """accuracy/AUC/precision/recall (binary, "Fallo" positive class) on the TEST split —
    key names unified with ml45_cereals_dnsl_critical_point_detection (both real training
    repos, a43-44-neurofuzzy-anomalias-fallas and a45-dnsl-cereals-deteccion-puntos-criticos,
    already use this exact naming in their own results.json). Renamed from the previous
    test_accuracy/test_auc/test_precision/test_recall so stats(mlflow_run_id=...) can
    overwrite the same keys it started from instead of adding new ones alongside them —
    modelo 43-45 audit, metrics unification."""
    macro_f1: float
    """Unweighted mean of F1 across Normal/Fallo classes, on the TEST split. Renamed from
    test_f1_macro for the same unification."""
    macro_recall: float
    """Unweighted mean of recall across Normal/Fallo classes, on the TEST split. Renamed
    from test_recall_macro."""
    decision_threshold: float
    """Probability cutoff for this run — the value on the VALIDATION split that maximizes
    F1, never on test. Previously this was hardcoded to the base model's DECISION_THRESHOLD
    constant (0.41) for every retrain; now it is recalibrated per-run and also logged to
    MLflow so stats(mlflow_run_id=...) can surface the real value for this run."""
    n_windows_train: int
    n_windows_val: int
    n_windows_test: int
    n_windows_total: int
    """Renamed from n_train/n_val/n_test to keep the n_windows_* prefix consistent with
    n_windows_total."""
    split_train_pct: float
    split_val_pct: float
    split_test_pct: float
    """The (fixed, non-configurable) train/val/test split ratios actually applied, derived
    from TRAIN_EXTERNAL_VAL_PCT/TRAIN_EXTERNAL_TEST_PCT — surfaced so the platform can show
    the split ratio next to n_windows_train/val/test instead of only raw counts."""
    mlflow_run_id: str | None = None
    """The MLflow run_id the artifacts were actually uploaded to (modelo 43-44 audit,
    Fase 3, Bug A). If the caller passed an empty mlflow_run_id, the backend creates a
    new run internally (see mlflow_utils.upload_artifacts_to_mlflow) — without this
    field, that new run_id was never returned, so the platform had no way to know a
    retrained model existed under a different run than the one it pre-created. None
    when the MLflow upload itself failed (see upload_warning)."""
    upload_warning: str | None = None
