"""Constants for ml18 — meat spatial price forecast (GRU, Keras/TensorFlow).

See inbox/a18/manifest.yaml for the full contract, metrics and known issues. Forecasts
PRECIO MEDIO KG one month ahead per (CCAA, Producto) combination, using spatial lag features
from neighboring CCAA (VECINOS_CCAA below, literal from the delivered src/main.py).
"""

MODEL_ID = "ml18-meat-spatial-price-forecast"
ARTIFACT_FOLDER_NAME = "ml18_meat_spatial_price_forecast"
VERSION = "1.0.0"
FRAMEWORK = "tensorflow/keras"

MODEL_FILENAME = "model.joblib"
X_SCALER_FILENAME = "x_scaler.json"
Y_SCALER_FILENAME = "y_scaler.json"

DATE_COL = "Fecha"
CCAA_COL = "CCAA"
PRODUCTO_COL = "Producto"
TARGET_COL = "PRECIO MEDIO KG"
OWN_PRICE_FEATURE = "PRECIO_MEDIO_KG_PROPIO"

RAW_REQUIRED_COLS = (
    DATE_COL, CCAA_COL, PRODUCTO_COL,
    "Poblacion", "RentaHogar", "CONSUMO X CAPITA", "PENETRACION (%)",
)

# Orden vinculante — igual que config/config.yaml::features.input_features del código entregado.
FEATURE_COLUMNS = (
    "CONSUMO X CAPITA",
    "PENETRACION (%)",
    "Poblacion",
    "RentaHogar",
    OWN_PRICE_FEATURE,
    "LAG_CONSUMO_VECINOS",
    "LAG_PRECIO_VECINOS",
)

LOOKBACK = 12

HIDDEN_UNITS = 96
DROPOUT = 0.2
DENSE_UNITS = 32

# src/main.py::VECINOS_CCAA, literal — matriz de vecindad fija (16 CCAA peninsulares + Baleares
# y Canarias, ambas sin vecinas por ser insulares: add_spatial_lags cae al valor propio como
# fallback para estas dos, documentado explícitamente en la memoria).
VECINOS_CCAA = {
    "ANDALUCIA": ["EXTREMADURA", "CASTILLA LA MANCHA", "MURCIA"],
    "ARAGON": ["CATALUNA", "CASTILLA LA MANCHA", "CASTILLA Y LEON", "LA RIOJA", "NAVARRA", "COMUNIDAD VALENCIANA"],
    "ASTURIAS": ["CANTABRIA", "CASTILLA Y LEON", "GALICIA"],
    "BALEARES": [],
    "CANARIAS": [],
    "CANTABRIA": ["ASTURIAS", "CASTILLA Y LEON", "PAIS VASCO"],
    "CASTILLA LA MANCHA": ["ANDALUCIA", "ARAGON", "CASTILLA Y LEON", "EXTREMADURA", "MADRID", "MURCIA", "COMUNIDAD VALENCIANA"],
    "CASTILLA Y LEON": ["ASTURIAS", "CANTABRIA", "PAIS VASCO", "LA RIOJA", "ARAGON", "CASTILLA LA MANCHA", "MADRID", "EXTREMADURA", "GALICIA"],
    "CATALUNA": ["ARAGON", "COMUNIDAD VALENCIANA"],
    "COMUNIDAD VALENCIANA": ["CATALUNA", "ARAGON", "CASTILLA LA MANCHA", "MURCIA"],
    "EXTREMADURA": ["ANDALUCIA", "CASTILLA LA MANCHA", "CASTILLA Y LEON"],
    "GALICIA": ["ASTURIAS", "CASTILLA Y LEON"],
    "LA RIOJA": ["CASTILLA Y LEON", "ARAGON", "NAVARRA", "PAIS VASCO"],
    "MADRID": ["CASTILLA Y LEON", "CASTILLA LA MANCHA"],
    "MURCIA": ["ANDALUCIA", "CASTILLA LA MANCHA", "COMUNIDAD VALENCIANA"],
    "NAVARRA": ["ARAGON", "LA RIOJA", "PAIS VASCO"],
    "PAIS VASCO": ["CANTABRIA", "CASTILLA Y LEON", "LA RIOJA", "NAVARRA"],
}

METRICS_REPORTED = {
    "dataset": "test_holdout (split cronologico 70/15/15, 11600 filas de test)",
    "test_mae": 1.0041,
    "test_rmse": 2.2113,
    "test_mape_pct": 9.9323,
    "test_r2": 0.8114,
    "kpi_contractual": "MAPE test <= 20%",
    "accepted_for_production": True,
    "baselines_test_mape_pct": {"naive_estacional_lag12": 14.1242, "media_historica_ccaa_producto": 20.2947},
}
