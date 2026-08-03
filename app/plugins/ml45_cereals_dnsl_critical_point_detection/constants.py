MODEL_ID = "ml45-cereals-dnsl-critical-point-detection"
ARTIFACT_FOLDER_NAME = "ml45_cereals_dnsl_critical_point_detection"

MODEL_FILENAME = "best_dnf_model.pt"
SCALER_FILENAME = "scaler.pkl"
XAI_BACKGROUND_FILENAME = "xai_background.npy"

FRAMEWORK = "pytorch"
VERSION = "1.0.0"

SEQUENCE_LENGTH = 240
SOLAPAMIENTO_BETA = 0.5
DEFAULT_THRESHOLD = 0.73

# data_generation.sensors en config.yaml del equipo de IA, mismo orden que
# expected_sensor_columns() del código original.
SENSOR_COLUMNS = [
    "plenum_temp",
    "exhaust_air_temp",
    "exhaust_air_humidity",
    "static_pressure",
    "burner_power",
    "fan_speed",
    "discharge_frequency",
    "grain_moisture_in",
    "ambient_temp",
    "ambient_humidity",
    "setpoint_temp",
]

TIMESTAMP_COLUMN = "timestamp"
ID_COLUMN = "cycle_id"
TARGET_COLUMN = "fault_name"

STATS_CREATION = ["mean", "std", "min", "max", "slope"]

NORMAL_TOKENS = [
    "0", "normal", "none", "ok", "healthy",
    "no failure", "no fallo", "sin falla", "sin fallo",
    "nofailure", "nofallo", "normal operation",
]

PARTIAL_NULL_MAX_RATIO = 0.10

# xai.pcc — catálogo + política de monitorización (config.yaml::xai del equipo de IA).
PCC_SUBSYSTEMS_CONFIG = [
    {"name": "humedad", "features": ["exhaust_air_humidity", "grain_moisture_in"]},
    {"name": "termico_transferencia", "features": ["plenum_temp", "exhaust_air_temp", "burner_power"]},
    {"name": "ventilacion_presion", "features": ["static_pressure", "fan_speed"]},
    {"name": "descarga_control", "features": ["discharge_frequency", "setpoint_temp"]},
    {"name": "contexto_operativo", "features": ["ambient_temp", "ambient_humidity"]},
]

PCC_CATALOG_RECORDS = [
    {
        "sub1": "humedad", "sub2": "termico_transferencia", "span": "final",
        "name": "PCC térmico-humedad tardío",
        "message": "Perfil crítico asociado al acoplamiento entre transferencia térmica y eliminación de humedad.",
        "recommendation": (
            "Revisar evolución de humedad del grano, temperatura de plenum, temperatura de salida y "
            "potencia del quemador. Validar que la transferencia térmica permite una evacuación adecuada de humedad."
        ),
    },
    {
        "sub1": "descarga_control", "sub2": "termico_transferencia", "span": "medio",
        "name": "PCC térmico-descarga intermedio",
        "message": "Perfil altamente discriminativo asociado a la interacción entre transferencia térmica y dinámica de descarga del material.",
        "recommendation": (
            "Comprobar que la regulación de descarga mantiene tiempos de residencia adecuados y no compromete "
            "la transferencia térmica ni la uniformidad del secado."
        ),
    },
    {
        "sub1": "descarga_control", "sub2": "termico_transferencia", "span": "final",
        "name": "Perfil ambiguo térmico-descarga tardío",
        "message": "Perfil frecuente asociado a la interacción entre transferencia térmica y control de descarga, con separación limitada entre normalidad y anomalía.",
        "recommendation": "Mantener seguimiento reforzado de la frecuencia de descarga, temperaturas características del proceso y potencia térmica aplicada.",
    },
    {
        "sub1": "descarga_control", "sub2": "ventilacion_presion", "span": "final",
        "name": "Perfil ambiguo descarga-ventilación tardío",
        "message": "Perfil asociado a posibles casos anómalos sobre la interacción entre descarga del material y condiciones de ventilación/presión.",
        "recommendation": "Monitorizar conjuntamente frecuencia de descarga, presión estática y condiciones de ventilación.",
    },
]

PCC_MONITOR_POLICY = {
    "normal_margin": 0.35,
    "critical_margin": 0.15,
    "min_support_catalog": 0.75,
    "min_subsystem_score": 0.05,
    "min_subsystem_variables": 1,
}

XAI_N_BACKGROUND = 64
XAI_TOP_RULES = 5
XAI_TOP_VARIABLES = 8
