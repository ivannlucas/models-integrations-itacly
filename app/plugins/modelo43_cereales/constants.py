"""Static configuration for the modelo43-cereales (cereal oven anomaly detector) plugin."""

MODEL_ID = "modelo43-cereales"
ARTIFACT_FOLDER_NAME = "modelo_43_cereales"

MODEL_FILENAME = "best_dnf_model.pt"
SCALER_FILENAME = "scaler.pkl"
XAI_BACKGROUND_FILENAME = "xai_background.npy"

FRAMEWORK = "pytorch/pandas/numpy/scikit-learn/shap"
VERSION = "1.0.0"

# Defaults from a43-44-neurofuzzy-anomalias-fallas/models/metrics/results.json
DECISION_THRESHOLD = 0.41
SEQ_LENGTH = 180
SOLAPAMIENTO_BETA = 0.5
ID_COLUMN = "cycle_id"
TIMESTAMP_COLUMN = "timestamp"
TARGET_COLUMN = "fault_name"
PARTIAL_NULL_MAX_RATIO = 0.10

SENSOR_COLUMNS = [
    "temp_zona1", "temp_zona2", "temp_zona3", "temp_salida_gases",
    "presion_camara", "presion_ventilacion", "potencia_kw", "flujo_gas",
    "humedad_relativa", "temp_ambiente", "setpoint_temp",
    "posicion_valvula", "velocidad_ventilador",
]
STATS_CREATION = ["mean", "std", "slope", "max", "min"]

# Training hyperparameters — verbatim from a43-44-neurofuzzy-anomalias-fallas/config/
# config.yaml (training: / data_processing.external_data_split:). Kept as named
# constants (not re-derived) so a config.yaml change in the training repo has an
# obvious, single place to update here too — see plugin.py::train() (modelo 43-44
# audit, Fase 5: decision_threshold parity with local training).
TRAIN_EXTERNAL_VAL_PCT = 13.3
TRAIN_EXTERNAL_TEST_PCT = 20.0
TRAIN_BATCH_SIZE = 256
TRAIN_SHUFFLE = False
TRAIN_LR = 0.00175
TRAIN_WEIGHT_DECAY = 0.00001
TRAIN_NUM_EPOCHS = 200
TRAIN_WARMUP_EPOCHS = 20
TRAIN_PATIENCE = 30
TRAIN_MIN_DELTA = 1e-4
TRAIN_GRAD_CLIP = 1.0
TRAIN_SCHEDULER_T_MAX = 200
TRAIN_SCHEDULER_ETA_MIN = 0.000009
TRAIN_THRESHOLD_SEARCH_POINTS = 101


# monitor_metric from config.yaml, hardcoded rather than re-implementing its generic
# AST-expression evaluator (config.yaml's formula is fixed for this deployment, not
# user-configurable) — used to select the best checkpoint during training.
def monitor_score(val_out: dict) -> float:
    return (
        0.65 * val_out["anomaly_f1"]
        + 0.25 * val_out["fuzzy_f1"]
        - 0.05 * val_out["dead_rules_ratio"]
        - 0.05 * val_out["alpha_entropy_mean"]
        - 0.05 * val_out["rule_corr_mean"]
    )


DNF_LOSS_KWARGS = {
    "anomaly_weight": 2.0,
    "aux_anom_dl_weight": 0.4,
    "aux_anom_fuzzy_weight": 0.95,
    "diversity_weight": 0.005,
    "alpha_entropy_weight": 0.01,
    "rule_structure_div_weight": 0.02,
    "rule_usage_balance_weight": 0.04,
    "reduction": "mean",
    "eps": 1e-8,
}

NORMAL_TOKENS = [
    "0", "normal", "none", "ok", "healthy",
    "no failure", "no fallo", "sin falla", "sin fallo",
    "nofailure", "nofallo", "normal operation",
]

# Defaults from a43-44-neurofuzzy-anomalias-fallas/models/artifacts/args.yaml — used to
# build a fresh model in train() and as a fallback if a checkpoint has no embedded model_cfg.
DEFAULT_MODEL_CFG = {
    "input_features": 13,
    "sequence_length": 180,
    "n_stats_features": 65,
    "lstm": {
        "hidden_size": 32,
        "num_layers": 2,
        "dropout": 0.30,
        "bidirectional": True,
        "embedding_dim": 32,
    },
    "fuzzy": {
        "n_mf": 5,
        "n_rules": 24,
        "train_membership_params": True,
        "train_rule_params": True,
        "temperature": 0.8,
        "t_norm": "product",
        "normalize_rules": True,
        "init_alpha": "sparse",
        "use_log_bias": False,
        "lambda_anomaly": 0.7,
    },
}

# Test-split metrics from a43-44-neurofuzzy-anomalias-fallas/models/metrics/results.json
# (test_metrics block) — key names unified with ml45_cereals_dnsl_critical_point_detection's
# own TEST_METRICS (both real training repos already use this exact naming), so the
# platform's "Atributos y métricas" panel shows the same labels for both models instead of
# two different legacy conventions (anomaly_* here vs accuracy/fallo_*/macro_* there).
TEST_METRICS = {
    "accuracy": 0.989,
    "fallo_auc": 0.9953633937559818,
    "fallo_precision": 0.8876404494382022,
    "fallo_recall": 0.9239766081871345,
    "macro_f1": 0.949802225840293,
    "macro_recall": 0.958453486136692,
}

# The metric keys stats(mlflow_run_id=...) is allowed to overwrite base.metrics with, once
# train() logs a real per-run value under each of these exact names — modelo 43-45 audit,
# metrics unification.
UNIFIED_METRIC_KEYS = (
    "accuracy", "fallo_auc", "fallo_precision", "fallo_recall",
    "macro_f1", "macro_recall", "decision_threshold",
)
