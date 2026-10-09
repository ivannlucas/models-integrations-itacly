MODEL_ID = "ml45-cereals-dnsl-critical-point-detection"
ARTIFACT_FOLDER_NAME = "ml45_cereals_dnsl_critical_point_detection"

MODEL_FILENAME = "best_dnf_model.pt"
SCALER_FILENAME = "scaler.pkl"
XAI_BACKGROUND_FILENAME = "xai_background.npy"

FRAMEWORK = "pytorch"
VERSION = "1.0.0"

SEQUENCE_LENGTH = 240
SOLAPAMIENTO_BETA = 0.5
DEFAULT_THRESHOLD = 0.73

# data_generation.sensors en config.yaml del equipo de IA, mismo orden que
# expected_sensor_columns() del código original.
SENSOR_COLUMNS = [
    "plenum_temp",
    "exhaust_air_temp",
    "exhaust_air_humidity",
    "static_pressure",
    "burner_power",
    "fan_speed",
    "discharge_frequency",
    "grain_moisture_in",
    "ambient_temp",
    "ambient_humidity",
    "setpoint_temp",
]

TIMESTAMP_COLUMN = "timestamp"
ID_COLUMN = "cycle_id"
TARGET_COLUMN = "fault_name"

STATS_CREATION = ["mean", "std", "min", "max", "slope"]

NORMAL_TOKENS = [
    "0", "normal", "none", "ok", "healthy",
    "no failure", "no fallo", "sin falla", "sin fallo",
    "nofailure", "nofallo", "normal operation",
]

PARTIAL_NULL_MAX_RATIO = 0.10

# xai.pcc — catálogo + política de monitorización (config.yaml::xai del equipo de IA).
PCC_SUBSYSTEMS_CONFIG = [
    {"name": "humedad", "features": ["exhaust_air_humidity", "grain_moisture_in"]},
    {"name": "termico_transferencia", "features": ["plenum_temp", "exhaust_air_temp", "burner_power"]},
    {"name": "ventilacion_presion", "features": ["static_pressure", "fan_speed"]},
    {"name": "descarga_control", "features": ["discharge_frequency", "setpoint_temp"]},
    {"name": "contexto_operativo", "features": ["ambient_temp", "ambient_humidity"]},
]

PCC_CATALOG_RECORDS = [
    {
        "sub1": "humedad", "sub2": "termico_transferencia", "span": "final",
        "name": "PCC térmico-humedad tardío",
        "message": "Perfil crítico asociado al acoplamiento entre transferencia térmica y eliminación de humedad.",
        "recommendation": (
            "Revisar evolución de humedad del grano, temperatura de plenum, temperatura de salida y "
            "potencia del quemador. Validar que la transferencia térmica permite una evacuación adecuada de humedad."
        ),
    },
    {
        "sub1": "descarga_control", "sub2": "termico_transferencia", "span": "medio",
        "name": "PCC térmico-descarga intermedio",
        "message": "Perfil altamente discriminativo asociado a la interacción entre transferencia térmica y dinámica de descarga del material.",
        "recommendation": (
            "Comprobar que la regulación de descarga mantiene tiempos de residencia adecuados y no compromete "
            "la transferencia térmica ni la uniformidad del secado."
        ),
    },
    {
        "sub1": "descarga_control", "sub2": "termico_transferencia", "span": "final",
        "name": "Perfil ambiguo térmico-descarga tardío",
        "message": "Perfil frecuente asociado a la interacción entre transferencia térmica y control de descarga, con separación limitada entre normalidad y anomalía.",
        "recommendation": "Mantener seguimiento reforzado de la frecuencia de descarga, temperaturas características del proceso y potencia térmica aplicada.",
    },
    {
        "sub1": "descarga_control", "sub2": "ventilacion_presion", "span": "final",
        "name": "Perfil ambiguo descarga-ventilación tardío",
        "message": "Perfil asociado a posibles casos anómalos sobre la interacción entre descarga del material y condiciones de ventilación/presión.",
        "recommendation": "Monitorizar conjuntamente frecuencia de descarga, presión estática y condiciones de ventilación.",
    },
]

PCC_MONITOR_POLICY = {
    "normal_margin": 0.35,
    "critical_margin": 0.15,
    "min_support_catalog": 0.75,
    "min_subsystem_score": 0.05,
    "min_subsystem_variables": 1,
}

XAI_N_BACKGROUND = 64
XAI_TOP_RULES = 5
XAI_TOP_VARIABLES = 8

# Training hyperparameters — verbatim from a45-dnsl-cereals-deteccion-puntos-criticos/
# config/config.yaml (training: / data_processing.external_data_split:). train()
# previously used a simplified "fine-tune the served weights, fixed 30 epochs, no
# threshold recalibration" shortcut — the real repo has no such concept, only
# scripts/train.py's full from-scratch pipeline (same template as ml43_cereals_dnsl_anomaly_fault_detection,
# confirmed near byte-identical trainer.py/metrics.py) — see plugin.py::train()
# (modelo 43-44-45 audit, Fase 5: decision_threshold parity with local training).
TRAIN_EXTERNAL_VAL_PCT = 13.3
TRAIN_EXTERNAL_TEST_PCT = 20.0
TRAIN_BATCH_SIZE = 128
TRAIN_LR = 0.002
TRAIN_WEIGHT_DECAY = 0.00001
TRAIN_NUM_EPOCHS = 200
TRAIN_WARMUP_EPOCHS = 10
TRAIN_PATIENCE = 20
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
        0.50 * val_out["anomaly_f1"]
        + 0.35 * val_out["fuzzy_f1"]
        - 0.05 * val_out["dead_rules_ratio"]
        - 0.05 * val_out["alpha_entropy_mean"]
        - 0.05 * val_out["rule_corr_mean"]
    )


DNF_LOSS_KWARGS = {
    "anomaly_weight": 2.0,
    "aux_anom_dl_weight": 0.35,
    "aux_anom_fuzzy_weight": 1.0,
    "diversity_weight": 0.0,
    "alpha_entropy_weight": 0.02,
    "rule_structure_div_weight": 0.04,
    "rule_usage_balance_weight": 0.05,
    "reduction": "mean",
    "eps": 1e-8,
}

# Base/served-model reference metrics, from
# a45-dnsl-cereals-deteccion-puntos-criticos/models/metrics/results.json (test_metrics
# block) — key names unified with ml43_cereals_dnsl_anomaly_fault_detection's own TEST_METRICS (both real training
# repos already use this exact naming), so the platform's "Atributos y métricas" panel shows
# the same labels for both models instead of two different legacy conventions. Overwritten
# key-by-key by stats(mlflow_run_id=...) once a real retrain logs its own values.
TEST_METRICS = {
    "accuracy": 0.9254166666666667,
    "fallo_auc": 0.9122309027777777,
    "fallo_precision": 0.9574468085106383,
    "fallo_recall": 0.65625,
    "macro_f1": 0.8669441348101212,
    "macro_recall": 0.8244791666666667,
}

# The metric keys stats(mlflow_run_id=...) is allowed to overwrite base.metrics with, once
# train() logs a real per-run value under each of these exact names — modelo 43-45 audit,
# metrics unification.
UNIFIED_METRIC_KEYS = (
    "accuracy", "fallo_auc", "fallo_precision", "fallo_recall",
    "macro_f1", "macro_recall", "decision_threshold",
)
