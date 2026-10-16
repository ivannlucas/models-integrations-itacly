"""Constants for ml24 — dairy spatial price forecast (CatBoost + calibrated baseline, h=1/3/6).

See inbox/a24/manifest.yaml for the full contract, metrics and known issues. Forecasts the
monthly average retail price of TOTAL LECHE LIQUIDA (EUR per kg or litre, MAPA household
consumption panel) per Comunidad Autonoma, 1, 3 and 6 months ahead. Every value below is copied
from the delivered config/config.yaml or the model metadata, never tuned here.
"""
# pylint: disable=line-too-long  # repo limit is 120 (.flake8), enforced by flake8

MODEL_ID = "ml24-dairy-spatial-price-forecast"
ARTIFACT_FOLDER_NAME = "ml24_dairy_spatial_price_forecast"
VERSION = "1.0.0"
FRAMEWORK = "catboost"

# Name the AI team gives the model inside its metadata/output (model_id/model_name columns).
ORIGINAL_MODEL_NAME = "spatial_price_model"

# ── Artifacts (flat in artifacts/<ARTIFACT_FOLDER_NAME>/, bit-for-bit copies of the delivery) ──
MODEL_MANIFEST_FILENAME = "model_manifest.json"
HORIZONS = (1, 3, 6)


def model_filename(horizon: int) -> str:
    """CatBoost model of one horizon (models/artifacts/spatial_price_model/h<h>/)."""
    return f"{ORIGINAL_MODEL_NAME}_h{horizon}.cbm"


def metadata_filename(horizon: int) -> str:
    """Selected features, baseline weights and ml_alpha of one horizon."""
    return f"{ORIGINAL_MODEL_NAME}_h{horizon}_metadata.json"


# Reference data the predictor rebuilds its runtime panels from (data/processed/...).
SERIES_FILENAME = "series_consumo_lacteos_ccaa.csv"
SPATIAL_WEIGHTS_FILENAME = "ccaa_border_percentage_matrix.csv"
EXOG_MONTHLY_FILENAME = "input_costs_monthly.csv"

# ── config/config.yaml :: data ────────────────────────────────────────────────────────────────
ANCHOR_PRODUCT = "TOTAL LECHE LIQUIDA"
CONTEXT_PRODUCTS = (
    "LECHE ESTERILIZADA",
    "LECHE ENTERA",
    "LECHE DESNATADA",
    "LECHE SEMIDESNATAD",
)
MODEL_PRODUCTS = (ANCHOR_PRODUCT, *CONTEXT_PRODUCTS)
INCLUDE_SPAIN = False
ANCHOR_PANEL_MIN_YEAR = 1999
CONTEXT_PANEL_MIN_YEAR = 2000
TRAIN_PANEL_MAX_YEAR = 2020        # anchor_panel.max_year = context_panel.max_year (training only)

TARGET_METRIC = "PRECIO MEDIO kg o litros"
TARGET_COL = "total_leche_liquida__precio_medio_kg_o_litros"
ENTITY_COL = "ccaa"

# Long-format series contract. es_reconstruido is required by build_product_panel even though
# the original validate_inference_input only lists the first six (see known_issues).
SERIES_REQUIRED_COLUMNS = ("anio", "mes", "ccaa", "producto", "metrica", "valor", "es_reconstruido")
MIN_HISTORY_MONTHS = 13            # lag_12 needs 13 months (memoria 8.4 / validate_inference_input)
MAX_VALIDATED_LAG_MONTHS = 24      # training.max_validated_lag_months

# src/utils/ccaa_codes.py::CCAA_CODE_TO_NAME, literal (ESP is listed by the original but the
# panels exclude T.ESPANA because include_spain is false, so it never yields a prediction).
CCAA_CODE_TO_NAME = {
    "AND": "ANDALUCIA",
    "ARA": "ARAGON",
    "NAV": "C. FORAL DE NAVARRA",
    "CAN": "CANARIAS",
    "CTB": "CANTABRIA",
    "CLM": "CASTILLA - LA MANCHA",
    "CYL": "CASTILLA Y LEON",
    "CAT": "CATALUNA",
    "MAD": "COMUNIDAD DE MADRID",
    "VAL": "COMUNITAT VALENCIANA",
    "EXT": "EXTREMADURA",
    "GAL": "GALICIA",
    "BAL": "ILLES BALEARS",
    "RIO": "LA RIOJA",
    "PVA": "PAIS VASCO",
    "AST": "PRINCIPADO DE ASTURIAS",
    "MUR": "REGION DE MURCIA",
    "ESP": "T.ESPANA",
}
SPAIN_NAME = "T.ESPANA"

# ── config/config.yaml :: training (shared by inference imputation and /train) ────────────────
TRAINING_CFG = {
    "entity_col": ENTITY_COL,
    "target_col": TARGET_COL,
    "task_type": "GPU",            # resolved at train time: GPU if CatBoost sees one, else CPU
    "devices": "0",
    "random_state": 24,
    "numpy_seed": 24,
    "publication_lag_months": 12,
    "max_validated_lag_months": MAX_VALIDATED_LAG_MONTHS,
    "baseline_calibration_months": 12,
    "weight_grid_step": 0.05,
    "loss_function": "MAE",
    "catboost_iterations": 300,
    "catboost_depth": 2,
    "catboost_learning_rate": 0.02,
    "catboost_l2_leaf_reg": 100.0,
    # config.yaml says `bootstrap_type: No`, which yaml.safe_load (YAML 1.1) parses as the boolean
    # False; build_model only passes bootstrap_type when truthy, so the AI team's models (the
    # served artifact included) were trained with CatBoost's DEFAULT bootstrap (MVS on CPU,
    # Bayesian on GPU), not "No". False reproduces that real behaviour (see known_issues).
    "bootstrap_type": False,
    "random_strength": 0.0,
    "missing_value_strategy": "median",
    "calibrate_ml_blend": True,
    "ml_blend_calibration_months": 12,
    "ml_alpha_grid": "0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1",
}
TRAIN_VALIDATION_YEAR = 2019       # validation_start/end = 2019-01-01 / 2019-12-01
TRAIN_TEST_YEAR = 2020             # test_start/end = 2020-01-01 / 2020-12-01

_EXOG_PATTERNS = ["exog__dairy", "exog__milk", "exog__ratio_cheese"]
# config/config.yaml :: horizons, literal.
HORIZONS_CFG = (
    {
        "horizon_months": 1, "panel_key": "anchor", "use_input_exog": True,
        "input_exog_patterns": _EXOG_PATTERNS, "exog_top_k": 6, "exog_stability_windows": 4,
        "exog_min_support": 2, "use_anchor_extra_metrics": True, "anchor_metrics_top_k": 4,
        "use_context_features": False, "context_top_k": 0,
        "panel_feature_stability_windows": 4, "panel_feature_min_support": 2,
    },
    {
        "horizon_months": 3, "panel_key": "anchor", "use_input_exog": True,
        "input_exog_patterns": _EXOG_PATTERNS, "exog_top_k": 6, "exog_stability_windows": 4,
        "exog_min_support": 2, "use_anchor_extra_metrics": False, "anchor_metrics_top_k": 0,
        "use_context_features": False, "context_top_k": 0,
        "panel_feature_stability_windows": 4, "panel_feature_min_support": 2,
    },
    {
        "horizon_months": 6, "panel_key": "context", "use_input_exog": True,
        "input_exog_patterns": _EXOG_PATTERNS, "exog_top_k": 6, "exog_stability_windows": 4,
        "exog_min_support": 2, "use_anchor_extra_metrics": False, "anchor_metrics_top_k": 0,
        "use_context_features": True, "context_top_k": 4,
        "panel_feature_stability_windows": 4, "panel_feature_min_support": 2,
    },
)
MLFLOW_ARTIFACT_PATH = "model"

# ── Reported metrics (models/metrics/spatial_price_model/metrics.csv, scope=global) ──────────
METRICS_REPORTED = {
    "dataset": "test 2020 (target_period_dt 2020-01..2020-12, 17 CCAA x 12 meses = 204 filas/horizonte)",
    "test": {
        "h1": {"mae": 0.013005, "smape": 1.870961, "mae_baseline": 0.014319, "smape_baseline": 2.059862,
               "accuracy_pct": 98.129},
        "h3": {"mae": 0.013768, "smape": 1.979459, "mae_baseline": 0.014839, "smape_baseline": 2.132642,
               "accuracy_pct": 98.021},
        "h6": {"mae": 0.014242, "smape": 2.045819, "mae_baseline": 0.015036, "smape_baseline": 2.158799,
               "accuracy_pct": 97.954},
    },
    "validation": {
        "h1": {"mae": 0.014850, "smape": 2.130136},
        "h3": {"mae": 0.013273, "smape": 1.906570},
        "h6": {"mae": 0.014537, "smape": 2.091440},
    },
    "kpi": "exactitud (100 - sMAPE) >= 80% -- cumple en los tres horizontes",
    "note": (
        "El artefacto servido se reentrenó con todo el histórico hasta 2020 (test incluido): estas "
        "métricas son de los modelos de etapa (train+val -> test), no del artefacto final."
    ),
}
