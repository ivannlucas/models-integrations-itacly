"""Constants for the ml34 dairy pasteurization energy GA plugin."""

MODEL_ID = "ml34-dairy-pasteurization-energy-ga"
ARTIFACT_FOLDER_NAME = "ml34_dairy_pasteurization_energy_ga"
MODEL_FILENAME = "mlp_predictor.pt"
MODEL_CONFIG_FILENAME = "model_config.json"
SCALER_X_FILENAME = "scaler_X.pkl"
SCALER_Y_FILENAME = "scaler_y.pkl"

FRAMEWORK = "pytorch+deap"
VERSION = "1.0.0"

# Feature order must match model_config.json features_in_order
FEATURES = ["T_in_leche", "F_flow", "T_servicio", "t_ciclo", "Delta_P"]
TARGETS = ["E_consumo", "T_out_leche"]

# Scenario (non-controllable) inputs required by optimize mode
SCENARIO_FEATURES = ["T_in_leche", "Delta_P", "t_ciclo"]

# ── GA hyperparameters (single-objective v4) — must match the original code
# (src/utils/constants.py GA_DEFAULT_CONFIG + src/predict/optimization.py) ──
GA_POP_SIZE = 150
GA_N_GEN = 15
GA_CXPB = 0.8
GA_MUTPB = 0.2
GA_SEED_BASE = 1
GA_BOUNDS = {
    "F_flow": (3500.0, 5500.0),     # L/h — pump operational limits
    "T_servicio": (76.0, 95.0),     # °C — boiler limits
}

# Food-safety constraint: legal HTST limit 72.0 °C + 0.3 °C margin for
# PT100 Class A uncertainty (IEC 60751) and FDA PMO tolerance.
T_OUT_MIN = 72.3       # °C
PENALTY_FACTOR = 10.0  # fitness penalty factor for infeasible solutions

# ── Training hyperparameters — from the original training code
# (model_config.json tuned params + src/main.py::train fixed settings) ──
TRAIN_LR = 0.0005
TRAIN_EPOCHS = 300
TRAIN_BATCH_SIZE = 128
TRAIN_PATIENCE = 15
TRAIN_SEED = 1
