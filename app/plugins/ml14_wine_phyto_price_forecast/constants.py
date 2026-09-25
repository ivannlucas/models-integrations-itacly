"""Constants for ml14 — wine phytosanitary price forecast (LSTM, PyTorch).

See inbox/a14/manifest.yaml for the full contract, metrics and known issues — in particular
model_status: this LSTM (selected by validation RMSE) does NOT beat the Drift baseline in the
final, correctly-audited test (RMSE 3.0168 vs 2.5406). It is served as-is because it is the
real, audited 2.0.0 artifact (not the retracted GRU numbers from the memoria/README), pending
a human decision on production exposure before the PR.
"""

MODEL_ID = "ml14-wine-phyto-price-forecast"
ARTIFACT_FOLDER_NAME = "ml14_wine_phyto_price_forecast"
VERSION = "1.0.0"
MODEL_VERSION = "2.0.0"  # versión del artefacto/modelo, per docs/MODEL_CARD.md — distinta de VERSION (API del plugin)
FRAMEWORK = "pytorch"

MODEL_FILENAME = "lstm_model.pt"
PREPROCESSING_FILENAME = "preprocessing.json"
FEATURES_FILENAME = "lstm_features.json"

# ── Feature engineering (fiel a src/data_processing/feature_engineering.py del código entregado) ──
DATE_COL = "date"
RAW_VALUE_COLS = (
    "PROTECCION_FITO",
    "CARBURANTES",
    "COPPER_EUR_TON",
    "GAS_EUR_MMBTU",
    "COIL_EUR_BARRIL",
    "DEXUSEU",
)
FEATURES_TO_PROCESS = (
    "COPPER_EUR_TON",
    "GAS_EUR_MMBTU",
    "DEXUSEU",
    "CARBURANTES",
    "PROTECCION_FITO",
)
LAGS = (4, 8, 12)
MOVING_AVERAGE_WINDOWS = (4, 8)
DIFF_MONTH_PERIOD = 4
EXPECTED_FREQUENCY_DAYS = 7
EXPECTED_WEEKDAY = 6  # domingo (pandas: lunes=0)

SEQ_LEN = 12
HORIZON_WEEKS = 16
_MAX_LAG = max(LAGS)
_ROLLING_LOOKBACK = max(MOVING_AVERAGE_WINDOWS) - 1
_FEATURE_LOOKBACK = max(_MAX_LAG, _ROLLING_LOOKBACK, DIFF_MONTH_PERIOD)
MIN_HISTORY_ROWS = _FEATURE_LOOKBACK + SEQ_LEN  # 12 + 12 = 24

TARGET_SERIES_COL = "PROTECCION_FITO"
DRIFT_COL = "PROTECCION_FITO_DIFF_MONTH"

# ── Modelo (LSTM 2.0.0 auditado — ver inbox/a14/manifest.yaml::model_status) ──
MODEL_NAME = "LSTM"
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3

# Métricas del test final auditado (docs/MODEL_CARD.md) — usadas en stats(), nunca en la
# reconstrucción de la predicción.
METRICS_REPORTED = {
    "dataset": "test_holdout_final",
    "split_method": "chronological_by_target_date",
    "canonical_seed": 42,
    "test_final_rmse_lstm": 3.0168249184,
    "test_final_rmse_drift": 2.5405822285,
    "test_final_mae_lstm": 2.2583,
    "test_final_mape_pct_lstm": 1.8608,
    "test_final_r2_lstm": 0.4969,
    "test_final_direction_acc_pct_lstm": 69.7,
    "validation_rmse_lstm": 0.9641,
    "accepted_for_production": False,
    "acceptance_criterion": "RMSE_LSTM_test_final < RMSE_Drift_test_final",
}
