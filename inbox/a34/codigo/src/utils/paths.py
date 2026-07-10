"""
Centralized project paths for DATAGIA.

All paths are computed from PROJECT_ROOT, which is the repository root
(the directory containing README.MD, src/, data/, models/, etc.).
"""

from pathlib import Path

# --- Project root (goes up from src/utils/ -> src/ -> project) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# --- Data ---
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
PREDICTIONS_DIR = DATA_DIR / "predictions"
IMAGES_DIR = DATA_DIR / "images"

# --- Models ---
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"
METRICS_DIR = MODELS_DIR / "metrics"

# --- Artifact files ---
MODEL_WEIGHTS_PATH = ARTIFACTS_DIR / "mlp_predictor.pt"
MODEL_CONFIG_PATH = ARTIFACTS_DIR / "model_config.json"
SCALER_X_PATH = ARTIFACTS_DIR / "scaler_X.pkl"
SCALER_Y_PATH = ARTIFACTS_DIR / "scaler_y.pkl"

# --- Data files ---
RAW_DATASET_PATH = RAW_DATA_DIR / "pasteurizacion_dataset_simulado.csv"
RAW_SETPOINTS_TABLE_PATH = RAW_DATA_DIR / "tabla_setpoints_operario.csv"
PROCESSED_DATASET_PATH = PROCESSED_DATA_DIR / "final_data_sim.csv"
TRAIN_SPLIT_PATH = SPLITS_DIR / "train.csv"
VAL_SPLIT_PATH = SPLITS_DIR / "val.csv"
TEST_SPLIT_PATH = SPLITS_DIR / "test.csv"

# --- Metrics files ---
TRAIN_METRICS_PATH = METRICS_DIR / "train_metrics.json"
BASELINE_METRICS_PATH = METRICS_DIR / "baseline_metrics.json"
GA_V3_REPORT_PATH = METRICS_DIR / "ga_v3_optimization_report.json"
GA_V4_REPORT_PATH = METRICS_DIR / "ga_v4_optimization_report.json"
EVAL_RT_REPORT_PATH = METRICS_DIR / "evaluation_rt_backtesting_report.json"

# --- Prediction files ---
GA_V3_RESULTS_PATH = PREDICTIONS_DIR / "ga_v3_optimization_results.csv"
GA_V4_RESULTS_PATH = PREDICTIONS_DIR / "ga_v4_optimization_results.csv"
EVAL_RT_CSV_PATH = PREDICTIONS_DIR / "evaluation_rt_hist_vs_ia.csv"
CONVERGENCE_CSV_PATH = PREDICTIONS_DIR / "ga_analisis_convergencia.csv"


def ensure_dirs() -> None:
    """Create all project directories if they do not exist."""
    for d in [
        RAW_DATA_DIR, PROCESSED_DATA_DIR, SPLITS_DIR, PREDICTIONS_DIR,
        IMAGES_DIR, ARTIFACTS_DIR, METRICS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
