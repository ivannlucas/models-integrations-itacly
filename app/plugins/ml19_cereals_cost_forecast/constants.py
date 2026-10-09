"""Constants for ml19 — cereals cost forecast (multi-horizon regression + classification
ensemble, Ridge/XGBoost/LogReg). See inbox/a19/manifest.yaml for the full contract, metrics
and known issues.

Architecture note (manifest.yaml, "DECISION DE ARQUITECTURA"): unlike other plugins, the client
does not supply a price history. The feature pipeline depends on 7 external data sources
(MAPA, ESYRCE, Copernicus/GEE climate, FAO, Yahoo Finance, ECB) and uses expanding() statistics
since 2006 that no API caller can reproduce. The delivered src/predict/predict.py itself reads
the latest (or a requested) row of an already-computed dataset_v7_fe.csv — it never takes raw
client data. This plugin bundles that dataset as a frozen reference artifact (see known_issues
in the manifest: it is NOT live-updated).
"""

MODEL_ID = "ml19-cereals-cost-forecast"
ARTIFACT_FOLDER_NAME = "ml19_cereals_cost_forecast"
VERSION = "1.0.0"
FRAMEWORK = "scikit-learn==1.8.0, xgboost==3.2.0"

TRAINING_META_FILENAME = "training_meta.json"
DATASET_FE_FILENAME = "dataset_v7_fe.csv"
DATE_COL = "date"

# src/utils/constants.py::MAPA_PUBLICATION_LAG_MONTHS, literal del codigo entregado.
MAPA_PUBLICATION_LAG_MONTHS = 3

METRICS_REPORTED = {
    "dataset": "test_walk_forward (2020-01..2025-09, periodo de prueba out-of-sample)",
    "regresion": {
        "h1m": {"modelo": "Ridge", "pearson_r": 0.604, "dir_accuracy_pct": 80.9, "sharpe": 2.61},
        "h2m": {"modelo": "XGBoost", "pearson_r": 0.559, "dir_accuracy_pct": 71.6, "sharpe": 1.66},
        "h3m": {"modelo": "XGBoost", "pearson_r": 0.611, "dir_accuracy_pct": 77.3, "sharpe": 2.71},
    },
    "clasificacion": {
        "h1m": {"modelo": "LogReg", "auc_roc": 0.802, "accuracy_pct": 68.1},
        "h2m": {"modelo": "XGBoost", "auc_roc": 0.778, "accuracy_pct": 71.0},
        "h3m": {"modelo": "XGBoost", "auc_roc": 0.842, "accuracy_pct": 78.3},
    },
    "criterios_exito_memoria": {
        "direction_accuracy_h1_min_pct": 60,
        "sharpe_h1_min": 1.0,
        "auc_roc_clasificacion_min": 0.65,
        "da_alta_confianza_min_pct": 65,
    },
    "kpi_contractual": (
        "Todos los umbrales superados en h=1m; h=2m y h=3m son senal orientativa "
        "(menor N independiente: ~34 y ~22 obs. respectivamente)."
    ),
}
