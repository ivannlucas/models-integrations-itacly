"""Constants for the ml38 wine machinery fault-detection plugin (a38, DNLS-VINO).

Every value is taken from the AI team's delivery (inbox/a38/codigo/, HEAD bfb3e13 = v1.6); the
source file is cited next to each block. See inbox/a38/manifest.yaml for the full traceability.
"""

MODEL_ID = "ml38-wine-machinery-fault-detection"
ARTIFACT_FOLDER_NAME = "ml38_wine_machinery_fault_detection"
VERSION = "1.0.0"

FRAMEWORK = "scikit-learn+keras"

# ── Delivered artifacts (codigo/models/artifacts/, served bit for bit) ───────────────────────
FAMILY_MODEL_FILENAME = "family_rf.pkl"
FAMILY_SCALER_FILENAME = "family_scaler.pkl"
SEVERITY_MODELS_FILENAME = "severity_models.pkl"
SEVERITY_SCALERS_FILENAME = "severity_scalers.pkl"
ARTIFACT_FILENAMES = (
    FAMILY_MODEL_FILENAME,
    FAMILY_SCALER_FILENAME,
    SEVERITY_MODELS_FILENAME,
    SEVERITY_SCALERS_FILENAME,
)

# ── Feature contract (src/data_processing/build_dataset.py::FEATURE_NAMES) ───────────────────
# The scalers were fitted on numpy arrays (no feature_names_in_): this ORDER is the contract.
FEATURE_NAMES = [
    # vibration time
    "vib_mean",
    "vib_std",
    "vib_max",
    "vib_min",
    "vib_rms",
    "vib_kurtosis",
    "vib_skewness",
    # vibration spectral
    "vib_spectral_energy",
    "vib_spectral_centroid",
    # vibration physical
    "vib_band_low",
    "vib_band_mid",
    "vib_band_high",
    "vib_amp_1x",
    "vib_amp_2x",
    "vib_ratio_2x_1x",
    # electrical time
    "ele_mean",
    "ele_std",
    "ele_max",
    "ele_min",
    "ele_rms",
    "ele_kurtosis",
    "ele_skewness",
    # electrical spectral
    "ele_spectral_energy",
    "ele_spectral_centroid",
]
LABEL_COLUMNS = ["family", "severity", "group_id"]   # src/main.py::train
FAMILY_COL = "family"
SEVERITY_COL = "severity"
GROUP_COL = "group_id"

HEALTHY_FAMILY = "healthy"
SEVERITY_LABELS = {0: "leve", 1: "significativo"}    # src/predict/predictor.py::run_inference
LABEL_NO_FAULT = "sin fallo"
LABEL_NO_SEVERITY_MODEL = "modelo no disponible"
SEVERITY_THRESHOLD = 0.5                             # prob > 0.5 → significativo (predictor.py)

# ── Signal processing (src/data_processing/preprocess.py) ────────────────────────────────────
FS_HZ = 20000
BAND_LOW_HZ = (0, 200)
BAND_MID_HZ = (200, 1000)
BAND_HIGH_HZ = (1000, 5000)
PEAK_TOL_HZ = 2
EPS = 1e-8
# get_rpm(): Motor-2 → 1470 rpm nominal, any other motor → 2950 rpm, scaled by speed %.
RPM_BASE_MOTOR_2 = 1470
RPM_BASE_OTHER = 2950

# Operating configurations documented in the NLN-EMP dataset (memoria Tabla 2) and the only ones
# present in the training data (input_manifest.csv). Raw inference outside them is rejected.
OPERATING_CONFIGURATIONS: dict[str, tuple[int, ...]] = {
    "Motor-2": (50, 75, 100),
    "Motor-4": (70,),
}
# Window length of the training samples (estimated by Parseval from the parquet, see manifest
# data.window_length_samples): 12 s of vibration and 15 s of current at 20 kHz. Energy, centroid
# (in FFT bins) and 1X/2X amplitudes scale with the length, so other lengths are off-scale.
TRAIN_VIB_WINDOW_SAMPLES = 240000
TRAIN_ELE_WINDOW_SAMPLES = 300000

# ── Training (config/config.yaml, consumed by src/training/*.py; = memoria §6.2) ────────────
TRAIN_RANDOM_STATE = 42
TRAIN_N_SPLITS = 5
TRAIN_RF_N_ESTIMATORS = 800
TRAIN_RF_MIN_SAMPLES_LEAF = 2
TRAIN_RF_CLASS_WEIGHT = "balanced"
TRAIN_BATCH_SIZE = 32
TRAIN_EPOCHS = 40
TRAIN_LEARNING_RATE = 0.001
TRAIN_SHUFFLE_BUFFER = 1000
TRAIN_HIDDEN_LAYERS = (64, 32)
TRAIN_DROPOUT = 0.3
TRAIN_ACTIVATION = "relu"
TRAIN_LAMBDA_PHYS = 0.3
TRAIN_LOW_BAND_INDEX = 9          # position of vib_band_low in FEATURE_NAMES
MIN_TRAIN_GROUPS = 2              # GroupKFold needs at least 2 groups

# ── Reported metrics (memoria v1.6 Tablas 9-14, models/metrics/*.json) ───────────────────────
REPORTED_METRICS = {
    "oof_family_accuracy": 0.6180,
    "oof_family_macro_f1": 0.5914,
    "cv_family_accuracy_mean": 0.6180,
    "cv_family_accuracy_std": 0.1133,
    "cv_family_macro_f1_mean": 0.6070,
    "cv_family_macro_f1_std": 0.1111,
    "oof_severity_accuracy": 0.5974,
    "oof_severity_macro_f1": 0.4334,
    "oof_severity_samples": 303,
    "oof_samples": 877,
    "split": "GroupKFold(5) por group_id — predicciones out-of-fold",
}
