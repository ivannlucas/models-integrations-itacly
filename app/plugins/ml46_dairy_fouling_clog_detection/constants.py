"""Static configuration for the ml46 dairy fouling/clog detection plugin."""

MODEL_ID = "ml46-dairy-fouling-clog-detection"
ARTIFACT_FOLDER_NAME = "ml46_dairy_fouling_clog_detection"

MODEL_FILENAME = "selected_model.pt"
FEATURE_ARTIFACTS_FILENAME = "feature_artifacts.json"
TRAINING_CONFIG_FILENAME = "training_config.json"
POLICY_THRESHOLDS_FILENAME = "policy_thresholds.json"

FRAMEWORK = "pytorch/pandas/numpy/scikit-learn"
VERSION = "1.0.0"
SCENARIO = "no_clock"  # fixed — the alternative "full" checkpoint is not served (manifest.constraints)
SEQ_LEN = 120
SEVERITY_COL = "Rf_m2K_W"

# manifest.yaml -> inputs.fixed (mínimo técnico común, verificado contra load_telemetry())
RAW_FIXED_COLUMNS = [
    "timestamp",
    "asset_id",
    "flow_kg_s",
    "pressure_in_kPa",
    "pressure_out_kPa",
    "dP_kPa",
    "Th_in_C",
    "Tc_in_C",
    "Th_out_C",
    "Tc_out_C",
    "Twall_C",
    "vibration_mm_s",
    "flow_sp_kg_s",
    "Th_sp_C",
    "Tc_sp_C",
    "protein_g_L_nominal",
    "fat_g_L_nominal",
    "solids_g_L_nominal",
    "Ca_mM_nominal",
    "PO4_mM_nominal",
    "pH_nominal",
]
# manifest.yaml -> inputs.recommended / inputs.optional
RAW_RECOMMENDED_COLUMNS = ["phase", "maintenance_active"]
RAW_OPTIONAL_COLUMNS = [
    "asset_family",
    "milk_type",
    "maintenance_type",
    "ambient_T_C",
    "ambient_RH_pct",
    "batch_thermal_history_factor",
    # NOT in memoria Tabla 9 (contrato de entrada productiva) but IS one of the 76 no_clock
    # model features (one-hot lastmaint_*). Without it, load_telemetry() defaults it to "none"
    # for every row, silently corrupting predictions for assets with real maintenance history.
    # See manifest known_issues.
    "last_maintenance_type",
]

# manifest.yaml -> training.required_columns.hard_required (load_telemetry raises without these)
TRAIN_HARD_REQUIRED_COLUMNS = ["timestamp", "asset_id", SEVERITY_COL]

# Output fields returned per scored window (manifest.yaml -> outputs.predict_inline)
PREDICT_OUTPUT_FIELDS = [
    "pred_severity",
    "pred_stage",
    "pred_stage_name",
    "p_stage0",
    "p_stage1",
    "p_stage2",
    "p_foul_h",
    "p_watch_fouling",
    "p_actionable_foul_h",
    "p_actionable_fouling",
    "p_clog_h",
    "pred_tte_foul_min",
    "pred_tte_clog_min",
    "pred_ttu_min",
]
