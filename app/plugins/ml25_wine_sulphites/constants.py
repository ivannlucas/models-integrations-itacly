"""Constants for the ml25 wine sulphites plugin."""

MODEL_ID = "ml25-wine-sulphites"
ARTIFACT_FOLDER_NAME = "ml25_wine_sulphites"
VERSION = "1.3.0"

# Batch predictions attach the full dose simulation (~40 points) to every row up to this many
# rows, so the monitoring panel can show each tank's trajectory (a winery sends 5-6 tanks a week).
# Larger batches keep only the first row's, to avoid multiplying the response size.
BATCH_TRAJECTORY_MAX_ROWS = 100

# Percentiles (p1/p99) of 'free sulfur dioxide' in the real training dataset
# (inbox/a25/codigo/data/white_wine.csv, 4898 filas) — fallback used by
# build_simulation_grid() when the loaded metadata.json doesn't carry a
# "simulation" block yet (i.e. the base artifact, trained before this field
# existed). Matches wine_quality.common.SimulationConfig(sim_free_p_low=1.0,
# sim_free_p_high=99.0) from the real delivered inference pipeline, which caps
# the dose-simulation grid to avoid extrapolating past the training
# distribution. A model retrained via train() stores its own percentiles in
# metadata["simulation"], which always takes precedence over these fallbacks.
BASE_FREE_SO2_P1 = 6.0
BASE_FREE_SO2_P99 = 81.0

BOUND_RF_MODEL_FILENAME = "bound_rf.pkl"
BOUND_XGB_MODEL_FILENAME = "bound_xgb.pkl"
QUALITY_RF_MODEL_FILENAME = "quality_rf.pkl"
QUALITY_XGB_MODEL_FILENAME = "quality_xgb.pkl"
METADATA_FILENAME = "metadata.json"
BOUND_XGB_JSON_FILENAME = "bound_xgb.json"
QUALITY_XGB_JSON_FILENAME = "quality_xgb.json"
