"""Constants for ml37 — predictive maintenance of vineyard presses/bottling equipment.

Faithful port of the delivered src/training/trainer.py, src/neurosymbolic/symbolic_engine.py and
src/data_processing/{build_schema_v1_tensor,adapt_external_tabular}.py (inbox/a37/codigo/).

IMPORTANT (see inbox/a37/manifest.yaml known_issues): the underlying dataset is NOT a real
vineyard — it is the public UCI "Condition Monitoring of Hydraulic Systems" dataset
(Helwig et al. 2015, DOI 10.24432/C5CW21), used by the AI team as a technical proxy for
hydraulic/electromechanical components (pump, valve, accumulator, cooling) functionally
comparable to presses, pumping, filtration and bottling equipment. Served variant: run_id
no_virtual_v1__rand_scool (the AI team's chosen reference configuration).
"""

MODEL_ID = "ml37-wine-press-predictive-maintenance"
ARTIFACT_FOLDER_NAME = "ml37_wine_press_predictive_maintenance"
VERSION = "1.0.0"
FRAMEWORK = "keras3(backend=torch) + scikit-learn (StandardScaler) + numpy/pandas"

# --- Artefactos (artifacts/ml37_wine_press_predictive_maintenance/, variante no_virtual_v1__rand_scool) ---
MODEL_FILENAME = "best_model.keras"
SCALER_FILENAME = "scaler.json"
CLASS_MAPPINGS_FILENAME = "class_mappings.json"
TRAIN_CONFIG_FILENAME = "train_config.json"

# --- Contrato schema_v1 (build_schema_v1_tensor.py) ---
TIMESTEPS = 600
F_SCHEMA = 40
# Orden vinculante de los 14 sensores físicos — slots 0-13 del tensor schema_v1. Debe coincidir
# EXACTAMENTE con class_mappings.json::feature_order_final[:14] y con scaler.json (mean/scale
# de longitud 40 se indexan por esta posición).
PHYSICAL_SENSORS = (
    "PS1", "PS2", "PS3", "PS4", "PS5", "PS6",
    "EPS1", "FS1", "FS2",
    "TS1", "TS2", "TS3", "TS4",
    "VS1",
)
# Slots 14-16: variables virtuales, forzadas a 0 en la configuración de producción (no_virtual).
VIRTUAL_SENSORS = ("CE", "CP", "SE")
# Slots 17-39: relleno técnico del contrato, siempre a 0 — nunca se piden al cliente.
N_RESERVED_SLOTS = F_SCHEMA - len(PHYSICAL_SENSORS) - len(VIRTUAL_SENSORS)

# --- Targets (4 heads de clasificación del modelo) ---
TARGET_NAMES = ("cooler", "valve", "pump", "accumulator")
# Código real UCI <-> índice de clase, en el orden que usa el softmax de cada head
# (class_mappings.json::class_order_by_target, idéntico a V1_CLASS_ORDERS de trainer.py).
CLASS_ORDERS = {
    "cooler": (3, 20, 100),
    "valve": (73, 80, 90, 100),
    "pump": (0, 1, 2),
    "accumulator": (90, 100, 115, 130),
}
TARGET_DESCRIPTIONS = {
    "cooler": "3=fallo/muy degradado, 20=eficiencia reducida, 100=eficiencia plena/OK",
    "valve": "73/80/90=condición degradada progresiva, 100=estado nominal",
    "pump": "0=nominal/OK, 1=degradación leve, 2=degradación severa",
    "accumulator": "90/100/115=presión/carga degradada, 130=estado nominal",
}

# --- Semáforo de incertidumbre (risk_* = 1 - max(softmax), NO es riesgo físico) ---
RISK_GREEN_MAX = 0.10
RISK_YELLOW_MAX = 0.25

# --- Capa neurosimbólica: umbrales físicos (symbolic_engine.py::BASELINE) ---
RULE_BASELINE = {
    "TS1_mean_max": 37.5,
    "TS1_mean_crit": 39.5,
    "dp_valve_min_warn": 1.5,
    "dp_valve_min_crit": 0.8,
    "pump_eff_min_warn": 0.0036,
    "pump_eff_min_crit": 0.0030,
    "PS1_mean_min_warn": 130.0,
    "PS1_mean_min_crit": 120.0,
    "PS1_std_max_warn": 4.0,
}
# Confianza por componente (symbolic_engine.py::MODEL_TRUST) — por debajo de TRUST_THRESHOLD,
# las alertas del modelo se degradan un nivel de severidad y se suprimen si la probabilidad de
# la clase predicha < MODEL_SUPPRESS_PROBA.
MODEL_TRUST = {"cooler": 0.97, "valve": 0.27, "pump": 0.64, "accumulator": 0.59}
TRUST_THRESHOLD = 0.70
MODEL_SUPPRESS_PROBA = 0.50
SEVERITY_RANK = {"INFO": 0, "WARNING": 1, "CRITICAL": 2, "EMERGENCY": 3}
RANK_SEVERITY = {v: k for k, v in SEVERITY_RANK.items()}

# --- Hiperparámetros de entrenamiento (trainer.py, train_config.json real) ---
TRAIN_SEED = 42
TRAIN_EPOCHS_MAX = 50
TRAIN_BATCH_SIZE = 256
TRAIN_EARLY_PATIENCE = 10
TRAIN_LEARNING_RATE = 0.001
TRAIN_MIN_DELTA = 1e-3
TRAIN_REDUCE_LR_FACTOR = 0.5
TRAIN_REDUCE_LR_PATIENCE = 3
TRAIN_MIN_LR = 1e-5
TRAIN_TEST_SIZE = 0.30  # luego 50/50 del remanente -> 70/15/15
TRAIN_LSTM_UNITS = 64
TRAIN_DENSE_UNITS = 64
TRAIN_DROPOUT = 0.2
# sample_weight: 0.5 si stable==1, 1.0 si stable==0 o no hay columna stable (trainer.py:647-657)
TRAIN_STABLE_WEIGHT = 0.5
TRAIN_UNSTABLE_WEIGHT = 1.0

# --- Métricas reales auditadas (models/metrics/no_virtual_v1__rand_scool/metrics_test.json,
# reproducidas de forma independiente en manifest-extraction, 331 ciclos de test) ---
METRICS_REPORTED = {
    "cooler": {"accuracy": 0.9668, "balanced_accuracy": 0.9667, "f1_macro": 0.9666},
    "valve": {"accuracy": 0.4683, "balanced_accuracy": 0.2804, "f1_macro": 0.2499},
    "pump": {"accuracy": 0.6495, "balanced_accuracy": 0.5677, "f1_macro": 0.5624},
    "accumulator": {"accuracy": 0.5498, "balanced_accuracy": 0.5452, "f1_macro": 0.5299},
    "support_test": 331,
    "kpi_negocio": (
        "Reducción del 20% del tiempo de inactividad no planificado -- NO medido ni demostrado "
        "con este dataset (declarado explícitamente en la memoria como objetivo pendiente de "
        "validación con datos reales de planta)."
    ),
}
