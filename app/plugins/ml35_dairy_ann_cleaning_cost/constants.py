"""Constants for the ml35 dairy ANN cleaning-cost plugin."""

MODEL_ID = "ml35-dairy-ann-cleaning-cost"
ARTIFACT_FOLDER_NAME = "ml35_dairy_ann_cleaning_cost"
MODEL_FILENAME = "model_ann.pt"
SCALER_X_FILENAME = "scaler_X.pkl"
SCALER_Y_FILENAME = "scaler_y.pkl"

FRAMEWORK = "pytorch+pygad"
VERSION = "1.0.0"

FEATURES = [
    "temp_entrada_leche",
    "temp_ambiente",
    "temp_setpoint_leche",
    "temp_proceso_leche",
    "temp_agua_servicio",
    "flujo_leche_lh",
    "horas_desde_limpieza",
    "presion_diferencial_bar",
]
TARGET_COL = "consumo_agua_l"

# GA hyperparameters — must match original training (deterministic: random_seed=42)
GA_NUM_GENERATIONS = 50
GA_NUM_PARENTS_MATING = 5
GA_SOL_PER_POP = 20
GA_RANDOM_SEED = 42
GA_GENE_SPACE = [
    {"low": 72.0, "high": 82.0},    # temp_setpoint_leche / temp_proceso_leche
    {"low": 75.0, "high": 92.0},    # temp_agua_servicio
    {"low": 2500.0, "high": 4500.0},  # flujo_leche_lh
]

# Baseline setpoints used for comparison in optimize mode
BASELINE_TEMP_LECHE = 80.0
BASELINE_TEMP_AGUA = 88.0
BASELINE_FLUJO = 3500.0

# Food-safety constraint (ISO 22000 / HTST)
PU_MIN = 13.0
TEMP_MIN_CELSIUS = 72.0
VOLUMEN_RETENCION_L = 16.0
Z_VALUE = 7.0
T_REF_CELSIUS = 72.0
