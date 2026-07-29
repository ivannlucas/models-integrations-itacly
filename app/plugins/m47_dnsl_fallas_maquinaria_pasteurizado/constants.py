import os

MODEL_ID = "m47-dnsl-fallas-maquinaria-pasteurizado"
ARTIFACT_FOLDER_NAME = "a47_dnsl_fallas_maquinaria_pasteurizado"

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
