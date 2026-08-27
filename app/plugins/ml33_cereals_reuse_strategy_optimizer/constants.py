"""Static configuration for the ml33 cereal reuse-strategy MILP optimizer plugin.

The deployed model (see optimizer.py) is a self-contained, deterministic mixed-integer
linear program (scipy.optimize.milp, HiGHS backend): no serialized weights, no reference
data files, no random seed. Same input -> same output, always. Ported from
src/predict/exact_optimizer.py in the original inbox/a33-cnp-cereals-neuroevolutivo-
reduccion-ambiental-residuos delivery (see inbox manifest.yaml for provenance).

NEAT (src/training/evolution.py, models/artifacts/winner_genome.pkl in the delivery) is
retained by the AI team only as a learned-policy benchmark, not the deployed engine — it
is intentionally NOT ported here (manifest known_issues).
"""

MODEL_ID = "ml33-cereals-reuse-strategy-optimizer"
ARTIFACT_FOLDER_NAME = "ml33_cereals_reuse_strategy_optimizer"

FRAMEWORK = "scipy.optimize.milp (HiGHS)"
VERSION = "1.0.0"

# Order matches the delivered code (exact_optimizer.py / inference.py STRATEGY_ORDER) —
# downstream reports and the MILP cost-vector layout depend on this exact order.
STRATEGY_ORDER: tuple[str, ...] = (
    "Biomass combustion",
    "Animal feed",
    "Composting",
    "Biochar",
)

# Deterministic temperature assumed per strategy in the emissions simulator. Never an
# input to the assignment decision itself — see strategy_emissions() in optimizer.py.
STRATEGY_TEMPERATURE_C: dict[str, float] = {
    "Animal feed": 60.0,
    "Composting": 60.0,
    "Biochar": 450.0,
    "Biomass combustion": 900.0,
}

ASSIGNMENT_SOURCE_EXACT = "exact_min_emissions"
ASSIGNMENT_SOURCE_CAPACITY_FALLBACK = "capacity_fallback"

# Defaults mirror config/pipeline_config.json ("inference" block) in the delivery — the
# "base validated inference regime" per the delivered README.md.
DEFAULT_LOTS_PER_DAY = 15
DEFAULT_ANIMAL_FEED_CAPACITY_T = 90.0
DEFAULT_COMPOSTING_CAPACITY_T = 140.0
DEFAULT_BIOCHAR_CAPACITY_T = 45.0
DEFAULT_BIOMASS_COMBUSTION_CAPACITY_T = 10000.0
DEFAULT_FALLBACK_STRATEGY = "Biomass combustion"

# Required columns for a batch CSV / a lot record (infer_dataframe() in the delivery).
REQUIRED_LOT_COLUMNS: tuple[str, ...] = (
    "generated_volume_tons",
    "moisture_pct",
    "subproduct_type",
    "season",
)
