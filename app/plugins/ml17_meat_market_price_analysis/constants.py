"""Constants for ml17 — meat market price analysis (Ridge, sklearn)."""

MODEL_ID = "ml17-meat-market-price-analysis"
ARTIFACT_FOLDER_NAME = "ml17_meat_market_price_analysis"
VERSION = "1.0.0"
FRAMEWORK = "sklearn"

MODEL_FILENAME = "ridge_official_v1_4.pkl"
LINE = "official_v1_4"

# Hiperparámetros del Ridge pipeline — idénticos a los usados por el equipo de IA para
# entrenar el artefacto fijo (config/official_v1_4.yaml: benchmark.tabular_models.ridge.alpha,
# inbox/a17/manifest.yaml: training.hyperparams). Usados por train() para reentrenar.
RIDGE_ALPHA = 1.0
IMPUTER_STRATEGY = "median"

FEATURE_COLUMNS = [
    "target_price_pigmeat_class_e_es",
    "eurostat_pigmeat_slaughter_tonnes_es",
    "eurostat_pigmeat_slaughter_tonnes_eu",
    "cereal_feed_barley_price_monthly",
    "cereal_feed_maize_price_monthly",
    "mapa_porcino_otras_razas_price_monthly",
    "month_sin",
    "month_cos",
]
