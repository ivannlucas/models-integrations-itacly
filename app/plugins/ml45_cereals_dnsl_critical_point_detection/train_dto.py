from pydantic import BaseModel, ConfigDict, Field


class TrainRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    data_path: str = Field(
        ...,
        description=(
            "Path to CSV with timestamp, cycle_id, the 11 sensor columns, and fault_name "
            "(ground truth, binarized internally via normal_tokens)."
        ),
    )
    mlflow_run_id: str = ""


class TrainResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    detail: str
    accuracy: float
    fallo_auc: float
    fallo_precision: float
    fallo_recall: float
    """accuracy/AUC/precision/recall (binary, "Fallo" positive class) on the TEST split —
    key names unified with modelo43_cereales (both real training repos,
    a43-44-neurofuzzy-anomalias-fallas and a45-dnsl-cereals-deteccion-puntos-criticos,
    already use this exact naming in their own results.json). Renamed from the previous
    bare accuracy/f1/auc fields, which used to leak into the platform's generic
    "Atributos y métricas" panel and accidentally trigger an unrelated hardcoded
    "Price fluctuation classifier" card there (ws-models.js duck-types on bare
    accuracy+f1 keys) — modelo 43-45 audit, metrics unification."""
    macro_f1: float
    """Unweighted mean of F1 across Normal/Fallo classes, on the TEST split. Renamed from
    test_f1_macro for the same unification."""
    macro_recall: float
    """Unweighted mean of recall across Normal/Fallo classes, on the TEST split. Renamed
    from test_recall_macro."""
    decision_threshold: float
    """Probability cutoff for this run — the value on the VALIDATION split that
    maximizes F1 (101-point grid search), never on test. Previously this plugin never
    calibrated a threshold at all (fine_tune() always reused the served model's fixed
    value) — now recalibrated per-run and logged to MLflow so
    stats(mlflow_run_id=...) and predict_batch/predict_inline can surface/use the
    real value for this run."""
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
    n_epochs: int
    """Epochs actually run before early stopping (or TRAIN_NUM_EPOCHS if it never
    triggered)."""
    mlflow_run_id: str | None = None
    upload_warning: str | None = Field(
        default=None,
        description="Non-fatal: training succeeded but the artifact upload to MLflow failed.",
    )
