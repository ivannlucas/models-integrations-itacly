import os

MODEL_ID = "ml47-dairy-dnsl-pasteurization-fault-detection"
ARTIFACT_FOLDER_NAME = "ml47_dairy_dnsl_pasteurization_fault_detection"

# ── Digital Twin ──────────────────────────────────────────────────────────────
# Aplica offset térmico a TS1/TS2 (desplaza ~20°C hacia arriba) para simular
# temperatura de planta real (65°C) cuando se usan datos del banco UCI (~45°C).
# En producción con datos reales de pasteurizador DEBE ser False.
# Activar solo para testing/desarrollo contra el dataset original UCI.
APPLY_DIGITAL_TWIN = os.getenv("APPLY_DIGITAL_TWIN", "false").lower() == "true"

MODEL_FILENAME = "neurosymbolic_cnn.pth"
SCALER_FILENAME = "scaler_cnn_dns.pkl"
FEATURE_COLUMNS_FILENAME = "feature_columns.pkl"
TS1_MEAN_FILENAME = "ts1_mean_train.pkl"

FRAMEWORK = "pytorch/scikit-learn"
VERSION = "1.0.0"

WINDOW_SIZE = 600
N_CLASSES = 3
SENSOR_COLUMNS = ["PS1", "PS3", "EPS1", "FS1", "TS1", "TS2", "VS1"]
COMPONENT_NAMES = ["Enfriador_Fouling", "Valvula_Switch", "Bomba_Leakage", "Acumulador_Gas"]
STATE_LABELS = {0: "SANO", 1: "WARNING", 2: "CRÍTICO"}

# ── /train — ported verbatim from the AI team's config/config.yaml ───────────
# (inbox/a47/codigo/config/config.yaml). Never tuned here.
# features.cols_targets — one column per component, values 0=Sano, 1=Warning, 2=Crítico.
TARGET_COLUMNS = ["Target_Fouling", "Target_Valvula", "Target_Bomba", "Target_Acumulador"]
# Names the previous plugin trainer required; still accepted so existing uploads keep working.
LEGACY_TARGET_COLUMNS = ["Fouling", "Valvula", "Bomba", "Acumulador"]

# mode="full" — training.* (Optuna optimum) used by src/training/trainer.py::train_model.
TRAIN_HYPERPARAMS = {
    "learning_rate": 0.0022243234786004373,
    "dropout_rate": 0.20219689010649033,
    "max_lambda": 3.7585293696964697,
    "epochs": 300,
    "warmup_epochs": 10,
    "ramp_up_epochs": 80,
    "patience": 15,
    "batch_size": 32,
}
# data.* — src/data_processing/preprocess.py::split_data / apply_digital_twin_and_augment.
NOISE_LEVEL = 0.2          # Gaussian noise (x column std) for the augmented copy of train cycles
SPLIT_TEST_SIZE_1 = 0.30   # train vs (val+test), by Cycle_ID
SPLIT_TEST_SIZE_2 = 0.50   # val vs test
RANDOM_STATE = 42          # splits + src/utils/reproducibility.py::seed_everything
DIGITAL_TWIN_TS1_TARGET = 65.0  # °C — thermal baseline of the digital twin offset

# mode="fine_tune" (default) — fine_tuning.* used by src/fine_tuning/fine_tuner.py::run_fine_tuning
# (frozen CNN backbone, only the 4 heads are recalibrated, LR = learning_rate / lr_divisor).
FINE_TUNE_HYPERPARAMS = {
    "epochs": 50,
    "patience": 7,
    "lr_divisor": 10,
    "batch_size": 32,
}
