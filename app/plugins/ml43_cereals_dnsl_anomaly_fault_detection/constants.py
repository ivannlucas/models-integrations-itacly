"""Static configuration for the ml43 cereal dryer DNF anomaly/fault detection plugin (CU43+CU44).

Sourced from inbox/a43/manifest.yaml (skill manifest-extraction) — hyperparameters and metrics
come from the delivered checkpoint (best_dnf_model.pt) + config.yaml/results.json, NOT the
memoria (see manifest known_issues: the memoria describes an older, superseded configuration).
"""

MODEL_ID = "ml43-cereals-dnsl-anomaly-fault-detection"
ARTIFACT_FOLDER_NAME = "ml43_cereals_dnsl_anomaly_fault_detection"

MODEL_FILENAME = "best_dnf_model.pt"
SCALER_FILENAME = "scaler.pkl"
XAI_BACKGROUND_FILENAME = "xai_background.npy"

FRAMEWORK = "pytorch/pandas/numpy/scikit-learn/shap"
VERSION = "1.0.0"

# manifest.yaml -> training.hyperparams (config.yaml/args.yaml del checkpoint entregado)
DECISION_THRESHOLD = 0.41000000000000003
SEQ_LENGTH = 180
SOLAPAMIENTO_BETA = 0.5
ID_COLUMN = "cycle_id"
TIMESTAMP_COLUMN = "timestamp"
TARGET_COLUMN = "fault_name"
PARTIAL_NULL_MAX_RATIO = 0.10

# manifest.yaml -> inputs.fixed
SENSOR_COLUMNS = [
    "temp_zona1", "temp_zona2", "temp_zona3", "temp_salida_gases",
    "presion_camara", "presion_ventilacion", "potencia_kw", "flujo_gas",
    "humedad_relativa", "temp_ambiente", "setpoint_temp",
    "posicion_valvula", "velocidad_ventilador",
]
STATS_CREATION = ["mean", "std", "slope", "max", "min"]

NORMAL_TOKENS = [
    "0", "normal", "none", "ok", "healthy",
    "no failure", "no fallo", "sin falla", "sin fallo",
    "nofailure", "nofallo", "normal operation",
]

# manifest.yaml -> training.hyperparams — fallback usado en train() y si un checkpoint no trae
# model_cfg embebido. Coincide exactamente con el checkpoint entregado (ver known_issues).
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

# manifest.yaml -> metrics_reported (models/metrics/results.json::test_metrics, real, no inventado)
TEST_METRICS = {
    "accuracy": 0.989,
    "macro_f1": 0.949802225840293,
    "macro_precision": 0.9415168937481585,
    "macro_recall": 0.958453486136692,
    "fallo_f1": 0.9054441260744985,
    "fallo_precision": 0.8876404494382022,
    "fallo_recall": 0.9239766081871345,
    "fallo_auc": 0.9953633937559818,
    "nofallo_f1": 0.9941603256060875,
    "nofallo_precision": 0.9953933380581148,
    "nofallo_recall": 0.9929303640862496,
    "nofallo_specificity": 0.9929303640862496,
    "best_epoch": 133,
    "n_epochs_trained": 163,
    "total_params": 51380,
}
