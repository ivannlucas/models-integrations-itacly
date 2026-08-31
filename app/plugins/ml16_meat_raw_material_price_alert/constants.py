"""Static configuration for the ml16 meat raw-material price alert plugin.

Values mirror the AI team's config/config.yaml and src/ code delivered in inbox/a16/codigo/
(see inbox/a16/manifest.yaml for provenance and known issues). Only the canonical h=4 pipeline
is covered here — the multi-horizon (h=1/h=2) artifacts from scripts/train_multi_horizon.py are
out of scope (see manifest known_issues).
"""

MODEL_ID = "ml16-meat-raw-material-price-alert"
ARTIFACT_FOLDER_NAME = "ml16_meat_raw_material_price_alert"
VERSION = "1.0.0"
FRAMEWORK = "xgboost/scikit-learn/pandas/numpy"

TARGETS = ("target_animales", "target_insumos")

# Fixed artifact filenames (same names as the delivered models/artifacts/)
MODEL_FILENAMES = {
    "target_animales": "model_target_animales.joblib",
    "target_insumos": "model_target_insumos.joblib",
}
SCALER_FILENAMES = {
    "target_animales": "scaler_target_animales.joblib",
    "target_insumos": "scaler_target_insumos.joblib",
}
BAGGING_FILENAMES = {
    "target_animales": "bagging_target_animales.joblib",
    "target_insumos": "bagging_target_insumos.joblib",
}
TRAIN_CONFIG_FILENAME = "train_config.json"

# Filenames used for user-retrained artifacts (never overwrite the fixed S3 artifacts above)
USER_MODEL_FILENAMES = {t: f"user_{name}" for t, name in MODEL_FILENAMES.items()}
USER_SCALER_FILENAMES = {t: f"user_{name}" for t, name in SCALER_FILENAMES.items()}
USER_BAGGING_FILENAMES = {t: f"user_{name}" for t, name in BAGGING_FILENAMES.items()}
USER_TRAIN_CONFIG_FILENAME = "user_train_config.json"

# predictor.py::_FEATURE_WARMUP_ROWS — filas que create_endogenous_features() elimina via
# dropna() antes de crear secuencias, dominado por precip_total_lag6 (shift(6), range(3,7)).
FEATURE_WARMUP_ROWS = 6

# config/config.yaml defaults — usados solo como fallback si faltan en train_config.json
# (nunca deberían faltar en un artefacto real).
DEFAULT_LOOKBACK = 3
DEFAULT_HORIZON = 4
DEFAULT_TEST_SIZE = 12
DEFAULT_N_SPLITS_CV = 5
DEFAULT_N_BAGGING = 12
DEFAULT_SEED = 42
DEFAULT_MANUAL_THRESHOLD_INSUMOS = 0.30
DEFAULT_USE_MANUAL_THRESHOLD = True

XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 2,
    "gamma": 0.1,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
}
LOGREG_PARAMS = {"C": 0.1, "max_iter": 1000, "solver": "lbfgs"}

# inputs.fixed / training.required_columns (inbox/a16/manifest.yaml) — mismo esquema que
# dataset_clasificacion_base.csv. 'month' es funcionalmente necesaria (create_endogenous_features
# la usa para month_sin/cos) aunque el código original nunca la derive de 'fecha'.
RAW_REQUIRED_COLUMNS = [
    "fecha", "month", "indice_animales", "indice_insumos",
    "precip_total", "precip_max", "wet_days", "wash_days", "animales_afectados",
]

# Test-split metrics of the delivered artifacts — fuente de verdad:
# models/metrics/hibrido_xgb_logreg_resumen.json, NO la memoria v1.5 (ver manifest known_issues).
METRICS_REPORTED = {
    "target_animales": {
        "modelo": "XGBoost",
        "umbral": 0.48,
        "validacion": {"Accuracy": 0.72, "Precision": 0.65, "Recall": 1.0, "F1": 0.788, "AUC": 0.833},
        "test": {"Accuracy": 0.833, "Precision": 0.833, "Recall": 0.833, "F1": 0.833, "AUC": 0.917},
    },
    "target_insumos": {
        "modelo": "LogReg",
        "umbral": 0.30,
        "validacion": {"Accuracy": 0.36, "Precision": 0.304, "Recall": 1.0, "F1": 0.467, "AUC": 0.917},
        "test": {"Accuracy": 0.667, "Precision": 0.429, "Recall": 1.0, "F1": 0.6, "AUC": 0.741},
    },
    "memoria_discrepancy_warning": (
        "La memoria v1.5 (Tabla 9) reporta umbral animales=0.46 y métricas de test distintas; "
        "los artefactos realmente entregados usan umbral=0.48 y las cifras de arriba — ver "
        "inbox/a16/manifest.yaml known_issues."
    ),
}
