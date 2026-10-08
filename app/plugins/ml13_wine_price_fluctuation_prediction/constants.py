"""Static configuration for the ml13 wine price fluctuation prediction plugin.

Values mirror the AI team's modules/common/config.py, modules/common/features.py and
modules/ML_models/models.py delivered in inbox/a13/codigo/ (see inbox/a13/manifest.yaml for
provenance and known issues). Only the production branch (tabular LogisticRegression / XGBoost,
models/prod/) is covered — the experimental GRU / GRU+LogReg ensemble (RNN_models/) has no
trained artifact and is excluded from production by the memoria.
"""

MODEL_ID = "ml13-wine-price-fluctuation-prediction"
ARTIFACT_FOLDER_NAME = "ml13_wine_price_fluctuation_prediction"
VERSION = "1.0.0"
FRAMEWORK = "scikit-learn/xgboost/pandas/numpy"

# Fixed artifact filenames (same names as the delivered models/prod/)
MODEL_FILENAME = "ml_model.pkl"
SCALER_FILENAME = "scaler.pkl"
FEATURE_SCHEMA_FILENAME = "feature_schema.json"
MODEL_CONFIG_FILENAME = "model_config.json"

# Filenames used for user-retrained artifacts (never overwrite the fixed S3 artifacts above)

# MLflow artifact sub-folder for user-retrained bundles
MLFLOW_ARTIFACT_PATH = "model"

# features.py — orden exacto de columnas de entrada del modelo (feature_schema.json)
FEATURE_COLUMNS = ["logret", "distsma12", "rsi14", "bollingerpos", "weeksin", "weekcos"]

# features.py::FeatureConfig
RSI_PERIOD = 14
SMA_SHORT_WINDOW = 4
SMA_LONG_WINDOW = 12
BOLLINGER_WINDOW = 20

# config.py::DataConfig
CAMPAIGN_COLUMN = "campaign"
WEEK_COLUMN = "week"
BULLETIN_COLUMN = "bulletin"
PRICE_COLUMN_CANDIDATES = ("preciotinto", "price_red", "precio_blanco", "price_white")

# config.py::MIN_INFERENCE_WEEKS — Bollinger(20) es la ventana dominante
MIN_INFERENCE_WEEKS = 20

# config.py::ModelConfig — definición del target y del protocolo de validación
RETURN_THRESHOLD = 0.025
TARGET_WINDOW = 4
TEST_SIZE = 24
N_FOLDS = 5
GAP = 4
RANDOM_SEED = 42

# Umbral de decisión fijo del código original (inference.py::_compute_test_metrics)
DECISION_THRESHOLD = 0.5

# config.py::ModelConfig — hiperparámetros de producción (models.py)
LOGREG_PARAMS = {"C": 1.0, "penalty": "l2", "class_weight": "balanced", "max_iter": 1000}
XGB_PARAMS = {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1}

# training.py::calculate_smart_score
SMART_SCORE_WEIGHTS = {"auc": 0.5, "f1": 0.3, "stability": 0.2}
STABILITY_STD_CAP = 0.15

# Métricas declaradas por el equipo de IA — memoria v1.10, Tablas 4, 5 y 6
METRICS_REPORTED = {
    "modelo": "LogisticRegression",
    "test_holdout": {
        "periodo": "2025-05-19..2025-10-27 (24 semanas, 8 positivos)",
        "AUC": 0.8438,
        "F1": 0.593,
        "Precision": 0.421,
        "Recall": 1.000,
        "Accuracy": 0.542,
        "smart_score": 0.7997,
    },
    "cv_walk_forward": {
        "AUC_mean": 0.6564,
        "AUC_std": 0.2704,
        "Accuracy": 0.659,
        "F1": 0.171,
        "Precision": 0.321,
        "Recall": 0.290,
        "smart_score": 0.3796,
    },
    "criterios_exito_memoria": {"AUC_min": 0.70, "F1_min": 0.50, "Recall_min": 0.65},
    "warning": (
        "En validación cruzada el modelo no alcanza los criterios de éxito de la memoria; el "
        "hold-out de test es, según la propia memoria, un límite superior bajo condiciones de "
        "mercado favorables (33,3% de positivos frente al 20% en CV). Ver inbox/a13/manifest.yaml."
    ),
}
