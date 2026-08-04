MODEL_ID = "ml28-meat-neuroevolutionary-raw-materials-prediction"
ARTIFACT_FOLDER_NAME = "ml28_meat_neuroevolutionary_raw_materials_prediction"

FRAMEWORK = "pandas/numpy (deterministic rules engine, no trained model — see constants below)"
VERSION = "1.0.0"

# config/platform_config.yaml del equipo de IA — constantes de heurística del motor de reglas
# real (src/cli/platform_run.py). No hay artefacto entrenado que cargar: estas constantes SON
# el "modelo".
PURCHASE_TRIGGER_GAP_SIGMOID_SCALE = 5.0
BASELINE_SAFETY_STOCK_FACTOR = 1.25
MAX_STOCKOUT_INCREASE_PCT = 5.0
ROUNDING_DECIMALS = 3

REQUIRED_COLUMNS = [
    "date",
    "raw_material_id",
    "current_inventory_tons",
    "expected_requirement_tons",
    "lead_time_days",
    "safety_coverage_days",
    "expected_yield_rate",
    "expected_waste_rate",
    "unit_purchase_cost",
    "shelf_life_days",
    "destination_profile",
]

# Catálogo descriptivo de destination_profile (platform_config.yaml::destination_profiles) —
# puramente informativo, no participa en ningún cálculo numérico.
DESTINATION_PROFILES = {
    "cooked_standard_line": "Stable cooked-line demand with medium shelf life.",
    "cured_long_horizon": "Longer curing cycle with higher lead-time exposure.",
    "fresh_short_shelf_life": "Short shelf-life fresh preparation profile.",
}
