"""Static configuration for the ml31 cereal residue-optimizer plugin (v2.0 LP).

The v2.0 model is a deterministic Linear Programming optimizer (PuLP/CBC). There
are NO serialized weights (.pkl/.pt): the "model" is the LP solver plus the
economic/agronomic reference data (JSON) and the historical dataset (CSV).
"""

MODEL_ID = "ml31-cereals-residue-optimizer"
ARTIFACT_FOLDER_NAME = "ml31_cereals_residue_optimizer"

# Reference-data artifacts (plain-text, deterministic — no serialized weights).
CROP_ECONOMICS_FILENAME = "crop_economics.json"   # prices (EUR/kg) + costs (EUR/ha) per crop
HARVEST_INDEX_FILENAME = "harvest_index.json"     # Harvest Index per crop/regime (+ stress variant)
DATASET_FILENAME = "dataset_cereal_lluvia_2009_2023.csv"  # historical JCyL 2009-2023 (no 2020)

FRAMEWORK = "pulp+cbc"
VERSION = "2.0.0"

# 10 crops => 20 decision variables (x_s[i], x_r[i]).
CROPS = [
    "Trigo duro",
    "Trigo semiduro y blando",
    "Cebada de 6 carreras",
    "Cebada de 2 carreras",
    "Avena",
    "Centeno",
    "Triticale",
    "Maíz",
    "Sorgo",
    "OTROS CEREALES",
]

# Success criteria (memoria Tabla 10 / config success_criteria defaults).
RESIDUE_REDUCTION_TARGET_PCT = 5.0
BENEFIT_PRESERVATION_TARGET_PCT = 100.0
MIN_BENEFIT_CHANGE_EUR = 0.0

# Water-stress threshold: spring rain < 100 mm activates the reduced (stress) HI.
WATER_STRESS_RAIN_MM = 100.0
