"""Static configuration for the meat-traceability incident-scoring plugin."""

MODEL_ID = "ml30-meat-traceability-detection"
ARTIFACT_FOLDER_NAME = "ml30_meat_traceability_detection"

MODEL_FILENAME = "model_state.pt"
PREPROCESSOR_FILENAME = "preprocessor.pkl"
MODEL_PAYLOAD_FILENAME = "model_payload.json"

FRAMEWORK = "pytorch/sklearn"
VERSION = "1.0.0"
DEFAULT_THRESHOLD = 0.5

NUMERIC_FEATURES = [
    "has_prev_event", "cold_start_lot", "stage_order_obs",
    "time_since_prev_lot_hours", "lot_stage_order_delta", "events_seen_before",
    "max_stage_order_seen_before", "stages_seen_count_before",
    "prior_sequence_anomaly_count", "sensor_temp_c", "sensor_ph",
    "sensor_weight_kg", "ts_hour", "ts_dayofweek",
    "sensor_temp_c_delta_from_prev", "sensor_ph_delta_from_prev",
    "sensor_weight_kg_delta_from_prev", "yield_pct_from_parent",
    "yield_delta_from_expected",
]
CATEGORICAL_FEATURES = [
    "stage", "prev_stage", "plant_line", "operator_shift",
    "process_route", "packaging_type", "trace_unit_type",
    "prev_trace_unit_type", "temp_sensor_location",
    "prev_temp_sensor_location", "ph_measurement_source",
    "prev_ph_measurement_source", "cold_room_id", "scale_id",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
