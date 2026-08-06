"""Static configuration for the ml9 cereal infestation sequence-classifier plugin.

Every value here traces back to inbox/a09/manifest.yaml (which in turn traces back to the delivered
code, artifacts and memoria). Do not invent defaults in this file.
"""

MODEL_ID = "ml9-cereals-infestation-sequence-classifier"
ARTIFACT_FOLDER_NAME = "ml9_cereals_infestation_sequence_classifier"

# ── Artifact filenames (manifest.artifacts) ───────────────────────────────────
MODEL_FILENAME = "final_winner.pt"                  # served checkpoint (GRU winner)
SCALER_FILENAME = "sequence_scaler.pkl"
BUNDLE_FILENAME = "model_bundle_metadata.json"
# Written by /train — the fixed S3 artifacts above are never overwritten by a user retrain.
USER_MODEL_FILENAME = "user_final_winner.pt"
USER_BUNDLE_FILENAME = "user_model_bundle_metadata.json"
USER_SCALER_FILENAME = "user_sequence_scaler.pkl"

FRAMEWORK = "pytorch/pandas/numpy/scikit-learn"
VERSION = "1.0.0"
TASK_TYPE = "timeseries_classification"

# ── Sequence contract (manifest.inputs.window; the bundle is authoritative at runtime) ────────
WINDOW_SIZE = 48        # 48 hourly observations per window
STRIDE = 12             # 12 h between consecutive windows
LABEL_MODE = "last"     # window label = label of its last step
N_FEATURES = 65         # engineered features expected by the checkpoint

# ── Column contract (manifest.inputs) ─────────────────────────────────────────
GROUP_COLUMN = "sample_id"
TIMESTAMP_COLUMN = "timestamp"
TARGET_COLUMN = "target"
SPLIT_TARGET_COLUMN = "target_global"
SENSOR_COLUMNS = ["co2_ppm", "temp_c", "ambient_rh_pct", "humidity_grain_pct"]
RAW_FIXED_COLUMNS = [GROUP_COLUMN, TIMESTAMP_COLUMN] + SENSOR_COLUMNS

# manifest.training.required_columns
TRAIN_HARD_REQUIRED_COLUMNS = RAW_FIXED_COLUMNS + [TARGET_COLUMN]
TRAIN_RECOMMENDED_COLUMNS = [SPLIT_TARGET_COLUMN]

# ── Class semantics (_vendor/sequential.py::LABEL_NAMES) ──────────────────────
CLASS_LABELS = {0: "sano", 1: "insectos", 2: "moho_critico"}
PROBA_FIELD_BY_CLASS = {0: "proba_sano", 1: "proba_insectos", 2: "proba_moho_critico"}

# manifest.outputs.predict_inline
PREDICT_OUTPUT_FIELDS = [
    "sample_id",
    "window_index",
    "timestamp_start",
    "timestamp_end",
    "pred_class",
    "pred_label",
    "proba_sano",
    "proba_insectos",
    "proba_moho_critico",
    "y_true",
]

# ── Reported hold-out metrics of the served GRU (manifest.metrics_reported) ───
HOLDOUT_METRICS = {
    "modelo_servido": "gru",
    "dataset": "holdout_test_split",
    "n_windows_test": 2220,
    "n_series_test": 60,
    "accuracy": 0.9414414414414415,
    "balanced_accuracy": 0.9452017539713365,
    "f1_macro": 0.9435867107634568,
    "precision_macro": 0.9422346311104768,
    "recall_macro": 0.9452017539713365,
    "log_loss": 0.18301969799876203,
    "por_clase": {
        "sano": {"precision": 0.9476, "recall": 0.9360, "f1": 0.9418, "support": 812},
        "insectos": {"precision": 0.9250, "recall": 0.9111, "f1": 0.9180, "support": 799},
        "moho_critico": {"precision": 0.9540, "recall": 0.9885, "f1": 0.9710, "support": 609},
    },
    "alternativa_no_servida_lstm_f1_macro": 0.93276291939956,
}

# manifest.metrics_reported.acceptance_thresholds — evaluation criteria, NOT inference rules
ACCEPTANCE_THRESHOLDS = {
    "f1_macro_min": 0.9,
    "recall_macro_min": 0.9,
    "accuracy_min": 0.9,
    "log_loss_max": 0.2,
    "class_recall_min": {"moho_critico": 0.95},
    "overall_passed": True,
}

SYNTHETIC_DATA_WARNING = (
    "Modelo entrenado y validado exclusivamente sobre el dataset sintético "
    "dataset_infestacion_cereales_sintetico (300 series, 144.000 observaciones horarias, generación "
    "interna). Las métricas acreditan la consistencia del pipeline en un escenario controlado, no el "
    "rendimiento sobre cereal real: la validación con datos reales de operación queda pendiente. "
    "Ver inbox/a09/manifest.yaml known_issues."
)
