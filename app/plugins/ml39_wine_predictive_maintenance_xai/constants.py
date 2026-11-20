"""Constants for ml39 — the XAI companion to ml37 (see inbox/a39/manifest.yaml).

a39 is NOT a new predictive model: it serves Integrated Gradients (always) and SHAP (opt-in)
explanations for the EXACT model, dataset and run_id (no_virtual_v1__rand_scool) that
ml37-wine-press-predictive-maintenance already serves. best_model.keras/scaler.json/
class_mappings.json are a39's own copy of ml37's artifacts (MD5-confirmed identical — see
manifest known_issues), not a live cross-plugin reference, following the same shape already
used in this repo for ml48 (XAI companion to ml47).
"""

MODEL_ID = "ml39-wine-predictive-maintenance-xai"
ARTIFACT_FOLDER_NAME = "ml39_wine_predictive_maintenance_xai"
VERSION = "1.0.0"
FRAMEWORK = "tensorflow (tf.GradientTape) + shap (GradientExplainer) + numpy/pandas"

# --- Artefactos (copia propia de los de ml37, mismo run_id no_virtual_v1__rand_scool) ---
MODEL_FILENAME = "best_model.keras"
SCALER_FILENAME = "scaler.json"
CLASS_MAPPINGS_FILENAME = "class_mappings.json"
# Nuevo artefacto propio de a39 (no viene en el entregable tal cual — ver manifest): muestra
# fija de ciclos de train para el fondo de SHAP, persistida una vez.
SHAP_BACKGROUND_FILENAME = "shap_background.npy"

# --- Contrato schema_v1 (idéntico al de ml37 — copia independiente, no importada) ---
TIMESTEPS = 600
F_SCHEMA = 40
PHYSICAL_SENSORS = (
    "PS1", "PS2", "PS3", "PS4", "PS5", "PS6",
    "EPS1", "FS1", "FS2",
    "TS1", "TS2", "TS3", "TS4",
    "VS1",
)
VIRTUAL_SENSORS = ("CE", "CP", "SE")
N_RESERVED_SLOTS = F_SCHEMA - len(PHYSICAL_SENSORS) - len(VIRTUAL_SENSORS)

# --- Targets (4 heads del modelo de ml37, mismo orden — confirmado contra HEAD_NAMES de
# xai_analysis.py y TARGET_NAMES de ml37/constants.py) ---
TARGET_NAMES = ("cooler", "valve", "pump", "accumulator")
CLASS_ORDERS = {
    "cooler": (3, 20, 100),
    "valve": (73, 80, 90, 100),
    "pump": (0, 1, 2),
    "accumulator": (90, 100, 115, 130),
}

# --- Integrated Gradients (xai_analysis.py::_integrated_gradients, perfil "full") ---
# steps=50 es el perfil "full" de config/config.yaml del entregable — el mismo que generó los
# golden_cases del manifest. NUNCA el perfil "audit" (steps=24): los valores no coincidirían con
# los golden cases ni con las cifras de la memoria.
IG_STEPS = 50

# --- SHAP (xai_analysis.py, perfil "full": shap_n_bg=80, seed=42) ---
SHAP_N_BACKGROUND = 80
SHAP_SEED = 42

# --- Métricas reales auditadas (heredadas de ml37: mismo modelo, mismo test set, ver manifest) ---
METRICS_REPORTED = {
    "cooler": {"accuracy": 0.9668, "balanced_accuracy": 0.9667, "f1_macro": 0.9666},
    "valve": {"accuracy": 0.4683, "balanced_accuracy": 0.2804, "f1_macro": 0.2499},
    "pump": {"accuracy": 0.6495, "balanced_accuracy": 0.5677, "f1_macro": 0.5624},
    "accumulator": {"accuracy": 0.5498, "balanced_accuracy": 0.5452, "f1_macro": 0.5299},
    "support_test": 331,
    "nota": (
        "Estas son las métricas de clasificación del modelo de ml37 (mismo modelo, mismo test "
        "set) — a39 no entrena ni evalúa un modelo propio, solo lo explica."
    ),
    "separabilidad_latente_silhouette": {
        "cooler": 0.5196, "valve": -0.0448, "pump": 0.0232, "accumulator": 0.0011,
    },
    "brier_score": {"cooler": 0.0177, "valve": 0.1512, "pump": 0.1427, "accumulator": 0.1377},
    "concordancia_ig_shap_spearman_r_perfil_full": {
        "cooler": 0.9165, "valve": 0.6484, "pump": 0.6044, "accumulator": 0.7978,
    },
    "kpi_negocio": (
        "Reducción del 20% del tiempo de inactividad no planificado — heredado de ml37, "
        "NOT_DEMONSTRATED: ningún análisis XAI mide ni puede medir este KPI con el dataset UCI."
    ),
}
