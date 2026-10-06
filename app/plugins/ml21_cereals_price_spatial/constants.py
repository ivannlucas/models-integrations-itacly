"""Constants for ml21 — ESP-CEREAL spatial cereal price prediction."""

MODEL_ID = "ml21-cereals-price-spatial"
ARTIFACT_FOLDER_NAME = "ml21_cereals_price_spatial"
VERSION = "1.0.0"
FRAMEWORK = "scikit-learn+xgboost"

MODEL_H1_REG = "datagia_best_h1_reg.joblib"
MODEL_H1_CLF = "datagia_best_h1_clf.joblib"
MODEL_H2_REG = "datagia_best_h2_reg.joblib"
MODEL_H2_CLF = "datagia_best_h2_clf.joblib"
MODEL_H3_REG = "datagia_best_h3_reg.joblib"
MODEL_H3_CLF = "datagia_best_h3_clf.joblib"
METADATA_FILENAME = "model_metadata.json"
# Base panel dataset (provincia x cereal x mes, con todas las columnas de ingenieria ya
# calculadas: lat_centroide, month_sin/cos, fase_*, lags de precios, clima, indices MAPA...).
# predict_inline solo recibe provincia+cereal+date del caller (manifest inputs.fixed) — el resto
# de las ~90 columnas que el modelo espera se buscan aqui, replicando run_single() del codigo
# original (src/predict/predict_v1.py). Sin este fichero, esas columnas se rellenarian con 0.
DATASET_FILENAME = "dataset_entrenamiento_final.csv"

VALID_HORIZONS = (1, 2, 3)

BLACKLIST = [
    "precio_provincial_lag_1",
    "precio_provincial_lag_2",
    "precio_provincial_lag_3",
    "precio_vecinos_media_lag1",
    "precio_nacional_base_ma3",
    "precio_nacional_base_ma6",
    "precio_nacional_base_vol3",
    "precio_nacional_base_vol6",
]

PROB_BULL = 0.65
PROB_BEAR = 0.35
RET_BULL = 0.015
RET_BEAR = -0.015
SPREAD_MIN = 0.01
CONFIDENCE_HIGH_UP = 0.60
CONFIDENCE_HIGH_DOWN = 0.40
TIMING_DELTA = 0.01
BENCHMARK_BREAK_DELTA = 0.01
MAPA_ADMIN_LAG = 3

GEO_RISK_DEFAULT_PROVINCES = {"Cuenca", "Lleida"}
