"""Constants for ml14 — wine phytosanitary price forecast (GRU, PyTorch).

See inbox/a14/manifest.yaml for the full contract, metrics and known issues. This is the
officially approved delivery (confirmed by the client after an earlier, unofficial local copy
with an internal data-leakage audit was initially used by mistake — see manifest known_issues
for that history). In this delivery GRU beats the Drift baseline on the reported test split.
"""

MODEL_ID = "ml14-wine-phyto-price-forecast"
ARTIFACT_FOLDER_NAME = "ml14_wine_phyto_price_forecast"
VERSION = "1.0.0"
FRAMEWORK = "pytorch"

MODEL_FILENAME = "gru_model.pt"
# Dataset bundled as reference data — necesario porque el CLI original (_build_scalers_and_
# features en src/predict/predictor.py) siempre reajusta (fit) los scalers de entrada/objetivo
# desde este mismo CSV en cada predicción, en vez de cargar un scaler ya serializado. No hay
# ningún preprocessing.json/features.json en esta entrega -- replicar el comportamiento real
# exige bundlear el propio dataset de entrenamiento.
REFERENCE_DATASET_FILENAME = "final_dataset_for_modeling.csv"

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

# ── Referencia (src/training/compare_models.py del código entregado): reconstrucción de las
# columnas horizonte y el split temporal usados para reajustar los scalers en cada predicción. ──
REFERENCE_DATE_COL = "FECHA"
TEST_RATIO = 0.2

# ── Modelo (GRU -- seleccionado por RMSE de test entre GRU/LSTM/XGBoost, ver manifest) ──
MODEL_NAME = "GRU"
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3

# Métricas del test reportado (models/metrics/model_comparison.json, split 80/20 único) — usadas
# en stats(), nunca en la reconstrucción de la predicción.
METRICS_REPORTED = {
    "dataset": "test_holdout (split temporal único 80/20, 211 observaciones)",
    "test_rmse_gru": 2.2873,
    "test_rmse_lstm": 2.2983,
    "test_rmse_xgboost": 3.3076,
    "test_rmse_drift": 2.5406,
    "test_rmse_naive": 3.8856,
    "test_mae_gru": 1.699,
    "test_mape_pct_gru": 1.3933,
    "test_r2_gru": 0.7108,
    "test_direction_acc_pct_gru": 77.7,
    "test_skill_score_gru": 0.6435,
    "n_seeds": 3,
    "accepted_for_production": True,
    "acceptance_criterion": "RMSE_GRU_test < RMSE_Drift_test (2.2873 < 2.5406 -- cumple)",
}
