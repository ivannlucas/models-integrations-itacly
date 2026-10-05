import os

MODEL_ID = "m48-dnsl-fallas-maquinaria-pasteurizado"
ARTIFACT_FOLDER_NAME = "a48_dnsl_fallas_maquinaria_pasteurizado"

# ── Digital Twin ──────────────────────────────────────────────────────────────
# Offset térmico que desplaza TS1/TS2 al baseline de 65 °C usando ts1_mean_train.
# SOLO para datos del banco de laboratorio UCI (aceite, ~30-45 °C). Con datos reales
# de pasteurizador DEBE ser False. Default de plataforma (env) y override por petición
# con el campo `apply_digital_twin`.
APPLY_DIGITAL_TWIN = os.getenv("APPLY_DIGITAL_TWIN", "false").lower() == "true"

MODEL_FILENAME = "neurosymbolic_cnn.pth"
SCALER_FILENAME = "scaler_cnn_dns.pkl"
FEATURE_COLUMNS_FILENAME = "feature_columns.pkl"
TS1_MEAN_FILENAME = "ts1_mean_train.pkl"
# Fondo de referencia para SHAP: 50 ciclos de test ya escalados (ver shap_background_meta.json).
SHAP_BACKGROUND_FILENAME = "shap_background.npy"
SHAP_BACKGROUND_META_FILENAME = "shap_background_meta.json"

FRAMEWORK = "pytorch/scikit-learn/shap"
VERSION = "1.0.0"

WINDOW_SIZE = 600
N_CLASSES = 3
SENSOR_COLUMNS = ["PS1", "PS3", "EPS1", "FS1", "TS1", "TS2", "VS1"]
COMPONENT_NAMES = ["Enfriador_Fouling", "Valvula_Switch", "Bomba_Leakage", "Acumulador_Gas"]
STATE_LABELS = {0: "SANO", 1: "WARNING", 2: "CRÍTICO"}

# ── XAI (valores de config/config.yaml y src/xai/ del equipo de IA) ──────────
HEAD_NAMES = ["Fouling", "Válvula", "Bomba", "Acumulador"]
CLASS_NAMES = ["Sano", "Warning", "Crítico"]
GRADCAM_LAYER = "features.17"
SAMPLING_RATE_HZ = 10.0
GRADCAM_PEAK_THRESHOLD = 0.7
SHAP_N_BACKGROUND = 50
SHAP_N_SAMPLES = 100
RANDOM_STATE = 42
RISK_WARNING = 0.5
RISK_CRITICAL = 0.8
SEVERITY_CRITICAL_RATIO = 0.10
SEVERITY_WARNING_RATIO = 0.20
DEFAULT_N_SAMPLES = 50
MAINTENANCE_ACTIONS_FILENAME = "maintenance_actions.yaml"

# ── Entrenamiento (config/config.yaml, óptimos Optuna del equipo de IA) ──────
TRAIN_HYPERPARAMS = {
    "learning_rate": 0.0022243234786004373,
    "dropout_rate": 0.20219689010649033,
    "max_lambda": 3.7585293696964697,
    "epochs": 300,
    "warmup_epochs": 10,
    "ramp_up_epochs": 80,
    "patience": 15,
    "batch_size": 32,
    "noise_level": 0.2,
    "test_size_1": 0.30,
    "test_size_2": 0.50,
    "random_state": 42,
}
TARGET_COLUMNS = ["Target_Fouling", "Target_Valvula", "Target_Bomba", "Target_Acumulador"]

# Métricas de referencia (models/metrics/test_metrics_confusion.csv, 331 ciclos de test)
REPORTED_METRICS = {
    "exact_match": 0.9879,
    "accuracy_fouling": 1.0,
    "accuracy_valvula": 1.0,
    "accuracy_bomba": 0.9940,
    "accuracy_acumulador": 0.9940,
}
