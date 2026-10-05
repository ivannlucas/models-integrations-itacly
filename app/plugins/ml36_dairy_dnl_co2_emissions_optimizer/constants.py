"""Constants for the ml36 dairy DNL CO2-emissions optimizer plugin."""

MODEL_ID = "ml36-dairy-dnl-co2-emissions-optimizer"
ARTIFACT_FOLDER_NAME = "ml36_dairy_dnl_co2_emissions_optimizer"
MODEL_FILENAME = "MLP_Final_Best_From_Search.pt"
MODEL_CONFIG_FILENAME = "final_model_config.json"
SCALER_X_FILENAME = "scaler_X.pkl"
SCALER_Y_FILENAME = "scaler_Y.pkl"
POLICY_FILENAME = "ga_policy_global.json"

FRAMEWORK = "pytorch+deap"
VERSION = "1.0.0"

# Feature order must match scaler_X / the delivered FEATURE_ORDER (11 columns)
FEATURE_ORDER = [
    "T_serv",
    "F_milk",
    "T_in",
    "Fat_perc",
    "Viscosity",
    "Regeneration_perc",
    "Delta_P",
    "t_ciclo",
    "Hydraulic_Resistance",
    "T_serv_margin",
    "Thermal_Intensity",
]
# The 8 base inputs a client provides; the other 3 are derived (see preprocessing.py)
BASE_FEATURES = [
    "T_serv", "F_milk", "T_in", "Fat_perc", "Viscosity",
    "Regeneration_perc", "Delta_P", "t_ciclo",
]
# Non-controllable scenario inputs required by optimize mode
CONTEXT_COLS = ["F_milk", "T_in", "Fat_perc", "Viscosity", "t_ciclo"]
# Genes decided by the GA, in individual order
GENE_ORDER = ["T_serv", "Delta_P", "Regeneration_perc"]

# MLP outputs — order is contractual (scaler_Y was fit on [T_out, CO2_emissions])
TARGETS = ["T_out", "CO2_emissions"]
T_OUT_IDX = 0
CO2_IDX = 1

# Constants hard-coded in the delivered feature engineering (ga_model_pipeline.py)
T_OUT_REFERENCE = 72.5
THERMAL_INTENSITY_DIVISOR = 853.6279

# Food-safety constraint: pasteurization threshold T_out >= 72.5 °C (soft penalty in the GA)
T_OUT_MIN = 72.5
PENALTY_FACTOR = 50.0

# GA gene bounds — config.yaml: ga_model.bounds (NOT the wider defaults in the code)
GA_BOUNDS = {
    "T_serv": (73.4, 84.31),
    "Delta_P": (0.6, 1.6),
    "Regeneration_perc": (86.9, 93.4),
}

# Adaptive (per-instance) GA — config.yaml: ga_model.realtime.adaptive_ga
GA_POP_SIZE = 16
GA_N_GEN = 14
GA_CXPB = 0.85
GA_MUTPB = 0.30
GA_ELITE_SIZE = 2
GA_MUTATION_ETA = 10.0
GA_MUTATION_INDPB = 0.6
GA_TOURNAMENT_SIZE = 3
GA_CX_ALPHA = 0.3
GA_BASE_SEED = 42
# Original row seed = GA_BASE_SEED + i, with i starting at 1 for the first row
DEFAULT_ROW_SEED = GA_BASE_SEED + 1
DECISION_MODES = ("static", "adaptive", "hybrid")
DEFAULT_DECISION_MODE = "hybrid"

# Training hyperparameters — config.yaml: training_model1.final_model + precomputed_best_config
TRAIN_LR = 0.0001
TRAIN_EPOCHS = 400
TRAIN_BATCH_SIZE = 128
TRAIN_PATIENCE = 30
TRAIN_SEED = 1
TRAIN_VAL_FRACTION = 0.15
