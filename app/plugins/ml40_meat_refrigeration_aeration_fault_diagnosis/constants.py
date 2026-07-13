"""Static configuration for the ml40 meat refrigeration/aeration fault diagnosis plugin.

Values mirror the AI team's config/config.yaml and src/ code delivered in inbox/a40/codigo/
(see inbox/a40/manifest.yaml for provenance and known issues).
"""

MODEL_ID = "ml40-meat-refrigeration-aeration-fault-diagnosis"
ARTIFACT_FOLDER_NAME = "ml40_meat_refrigeration_aeration_fault_diagnosis"

FRAMEWORK = "scikit-learn/pandas/numpy"
VERSION = "1.0.0"

SYSTEMS = ("refrigeracion", "aireado")

# Artifact filenames per system (same names as the delivered models/artifacts/)
MODEL_FILENAMES = {
    "refrigeracion": "refrigeracion_model.pkl",
    "aireado": "aireado_model.pkl",
}
SCALER_FILENAMES = {"refrigeracion": "refrigeracion_scaler.pkl"}  # aireado has no scaler by design
THRESHOLDS_FILENAMES = {
    "refrigeracion": "refrigeracion_thresholds.yaml",
    "aireado": "aireado_thresholds.yaml",
}
STATS_FILENAMES = {
    "refrigeracion": "refrigeracion_stats.yaml",
    "aireado": "aireado_stats.yaml",
}

# Filenames used for user-retrained artifacts (never overwrite the fixed S3 artifacts above)
USER_MODEL_FILENAMES = {
    "refrigeracion": "user_refrigeracion_model.pkl",
    "aireado": "user_aireado_model.pkl",
}
USER_SCALER_FILENAMES = {"refrigeracion": "user_refrigeracion_scaler.pkl"}
USER_STATS_FILENAMES = {
    "refrigeracion": "user_refrigeracion_stats.yaml",
    "aireado": "user_aireado_stats.yaml",
}

# config.yaml -> {system}.mapping (fault_id -> class name)
CLASS_MAPPINGS = {
    "refrigeracion": {
        0: "NORMAL",
        1: "COND_FOUL_MILD",
        2: "COND_FOUL_SEVERE",
        3: "EVAP_FAN_DEG",
        4: "EVAP_FAN_FAIL",
        5: "UNDERCHARGE_MILD",
        6: "UNDERCHARGE_SEVERE",
        7: "OVERCHARGE",
        8: "SENSOR_DRIFT_PLUS",
        9: "SENSOR_DRIFT_MINUS",
        10: "COMP_INEFFICIENCY",
        11: "NON_CONDENSABLES",
        12: "UNDERCHARGE_AND_COND_FOUL",
    },
    "aireado": {
        0: "NORMAL",
        1: "ENCOSTRAMIENTO",
        2: "SATURACION_HIELO",
        3: "FALLO_VENTILADOR",
    },
}

# config.yaml -> {system}.drop_cols + metadata always removed before inference
# (src/predict/predictor.py::run_inference)
DROP_COLS = {
    "refrigeracion": ["fault", "run_id", "fault_id", "time_min", "fault_numeric",
                      "T_cond_sat", "T_cab_meas", "P_suc_bar"],
    "aireado": ["run_id", "time_min", "fault_id", "fault"],
}
METADATA_COLS = ["run_id", "time_min", "fault_id", "fault", "fault_numeric", "prediction", "confidence"]

# Raw sensor columns required per system (manifest inputs.fixed; run_id/time_min listed apart)
RAW_INPUT_COLUMNS = {
    "refrigeracion": [
        "T_amb", "T_set", "T_cab", "T_evap_sat", "T_cond_sat", "P_suc_bar", "P_dis_bar",
        "N_comp_Hz", "SH_K", "P_comp_W", "Q_evap_W", "COP", "frost_level", "T_cab_meas",
        "valve_open", "door_open", "defrost_on",
    ],
    "aireado": [
        "Kg_embutido", "T_amb", "T_set", "N_fan_Hz", "RH_cab", "T_cab", "T_evap_sat", "P_comp_W",
    ],
}
CYCLE_COLUMNS = ["run_id", "time_min"]
TARGET_COLUMN = "fault_id"

# Engineered-input marker: if present, the CSV already went through feature engineering
# (equivalent to data/splits/{system}_test.csv) and the pipeline must NOT re-engineer it.
ENGINEERED_MARKER = {
    "refrigeracion": "Pdis_instability_20",
    "aireado": "Encostramiento_Risk",
}

# Column sets that unambiguously identify each system in an input CSV
SYSTEM_SIGNATURE = {
    "refrigeracion": {"P_dis_bar", "P_suc_bar", "T_cond_sat"},
    "aireado": {"RH_cab", "N_fan_Hz", "Kg_embutido"},
}

# Minimum minutes of history per cycle for a reliable diagnosis (memoria section 9:
# refrigeration lags reach 100 min + initial dropna; aeration lags reach 60 min)
MIN_HISTORY_MINUTES = {"refrigeracion": 100, "aireado": 60}

# Health monitoring (config.yaml). The original implementation keeps a rolling window of the
# last 50 confidences in logs/monitorization_{system}.csv; the plugin computes the status
# statelessly on the current request's mean confidence instead (see manifest known_issues).
HEALTH_THRESHOLD_PCT = 75.0

# config.yaml -> {system}.model_params (manifest training.hyperparams)
MODEL_PARAMS = {
    "refrigeracion": {
        "n_estimators": 200,
        "min_samples_leaf": 20,
        "max_features": "sqrt",
        "max_depth": 20,
        "random_state": 42,
        "n_jobs": -1,
    },
    "aireado": {
        "n_estimators": 200,
        "min_samples_split": 5,
        "min_samples_leaf": 4,
        "max_depth": 20,
        "bootstrap": False,
        "random_state": 42,
        "n_jobs": -1,
    },
}

# Binary columns excluded from the refrigeration StandardScaler (refrig_trainer.py)
REFRIGERACION_BINARY_COLS = ["defrost_on", "door_open"]

# Test-split metrics of the delivered artifacts (manifest metrics_reported; final
# RF + neurosymbolic rules + per-run vote configuration)
METRICS_REPORTED = {
    "aireado": {
        "dataset": "data/splits/aireado_test.csv (6.000 filas, 60 ciclos)",
        "accuracy": 1.00,
        "f1_macro": 1.00,
        "precision_macro": 1.00,
        "recall_macro": 1.00,
        "per_run_accuracy": 1.00,
    },
    "refrigeracion": {
        "dataset": "data/splits/refrigeracion_test.csv (370.500 filas, 260 ciclos)",
        "accuracy": 0.94,
        "f1_macro": 0.94,
        "precision_macro": 0.94,
        "recall_macro": 0.94,
        "per_run_accuracy": 0.95,
        "problem_classes": "COND_FOUL_SEVERE (f1=0.62) y NON_CONDENSABLES (f1=0.63) se confunden entre sí",
    },
    "synthetic_data_warning": (
        "Dataset de refrigeración simulado (Kaggle, refrigerador de propósito general) y de "
        "aireado 100% sintético — métricas sin validar contra datos reales de planta; ver "
        "inbox/a40/manifest.yaml known_issues."
    ),
}
