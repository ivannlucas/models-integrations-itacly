"""Static configuration for the ml26 wine sulfite (GRU + PSO) 72 h forecast plugin.

Every value here is traceable to inbox/a26/manifest.yaml (and from there to the AI team's code,
artifacts or memoria).
"""

MODEL_ID = "ml26-wine-sulfite-gru-pso-forecast"
ARTIFACT_FOLDER_NAME = "ml26_wine_sulfite_gru_pso_forecast"
VERSION = "1.0.0"
FRAMEWORK = "pytorch/pandas/numpy"
TASK_TYPE = "timeseries_regression_multitask"

# Fixed artifact delivered by the AI team (pickled SequenceBundle) + selection registry
MODEL_FILENAME = "gru_pso.pkl"
BEST_MODEL_FILENAME = "best_model.json"

# Format used for user fine-tuned models (local + MLflow). Safe to load: tensors via
# torch.load(weights_only=True) + plain JSON, no pickle.
USER_STATE_FILENAME = "model_state.pt"
USER_META_FILENAME = "bundle_meta.json"
USER_TRAINED_SUBDIR = "user_trained"
MLFLOW_ARTIFACT_PATH = "model"

# Mirror of the preprocessing block of inbox/a26/codigo/config/config.yaml (what
# build_feature_frame reads)
PIPELINE_CONFIG = {
    "experiment": {"seed": 42},
    "data": {"train_split": 0.7, "val_split": 0.15},
    "preprocessing": {
        "sequential_window": 24,
        "sequential_stride": 2,
        "decision_horizon_hours": 72,
        "target_followup_steps": 36,
        "categorical_columns": ["wine_type", "stage"],
        "categorical_levels": {
            "wine_type": ["red", "rose", "white"],
            "stage": ["fermentation", "must", "pre_bottling", "stabilization", "storage"],
        },
    },
}
WINDOW = 24
STRIDE = 2
STEP_HOURS = 2.0
TRAINING_SEED = 7
SPLIT_SEED = 42

TARGET_SO2 = "future_free_sulfite_72h"
TARGET_RISK = "underprotection_risk_72h"
TARGET_NAMES = [TARGET_SO2, TARGET_RISK]

WINE_TYPES = ["red", "rose", "white"]
STAGES = ["must", "fermentation", "stabilization", "storage", "pre_bottling"]

# manifest inputs.lot / inputs.readings
LOT_COLUMNS = ["lot_id", "wine_type", "volume_l", "ambient_temp_c"]
READING_REQUIRED_COLUMNS = [
    "lot_id",
    "stage",
    "temperature_c",
    "dissolved_oxygen_mg_l",
    "density_g_ml",
    "co2_g_l",
    # Mandatory by decision (manifest KI-01): the audited model needs it and the AI team's fallback
    # (observed-history approximation) triples the error. With it, the original preprocessing
    # reproduces the audited windows exactly.
    "stage_progress",
]
READING_OPTIONAL_COLUMNS = [
    "timestamp",
    "timestamp_index",
    "ph_lab",
    "free_sulfite_lab",
    "total_sulfite_lab",
    "dose_mg_l",
    "actual_dose_mg_l",
]

# manifest training.required_columns.hard_required
TRAIN_HARD_REQUIRED_COLUMNS = [
    "lot_id",
    "timestamp_index",
    "wine_type",
    "volume_l",
    "ambient_temp_c",
    "stage",
    "temperature_c",
    "dissolved_oxygen_mg_l",
    "density_g_ml",
    "co2_g_l",
    "free_sulfite_mg_l",
    "stage_progress",
]

# Risk bands — memoria v2.5 §7.1 (== notebooks/EDA/eda.ipynb::risk_band)
RISK_BAND_LOW_MAX = 0.33
RISK_BAND_MEDIUM_MAX = 0.66

# manifest metrics_reported (test hold-out, synthetic data) + operational_mode_impact
REPORTED_METRICS = {
    "dataset": "test_hold_out (90 lotes sintéticos, 3096 ventanas)",
    "future_free_sulfite_72h_rmse": 1.9765,
    "future_free_sulfite_72h_mae": 1.5082,
    "underprotection_risk_72h_rmse": 0.0642,
    "underprotection_risk_72h_mae": 0.0429,
    "overall_rmse": 1.3983,
    "overall_mae": 0.7755,
    "so2_within_3mg_l": 0.876,
    "recall_riesgo_alto": 0.925,
    "precision_riesgo_alto": 0.939,
    "synthetic_data_warning": (
        "Dataset, targets y métricas proceden del simulador del proyecto (no existe dataset "
        "público de este tipo: son datos propios de bodega); validación con históricos reales "
        "pendiente. "
        "El score de riesgo no es una probabilidad calibrada."
    ),
}
