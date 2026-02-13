"""
Configuration module for the wine price forecasting project.

This module centralizes all high-level configuration used by the codebase:
- Project paths (data directories, model artifact directories).
- Data configuration (column names, raw filename).
- Model configuration (lookback window, target horizon, thresholds).
- Basic hyperparameters for Logistic Regression and GRU.
Keeping these values here makes it easy to change behavior without editing
the training or inference logic.
"""

from pathlib import Path
from dataclasses import dataclass

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_PROD_DIR = MODELS_DIR / "prod"

OUTPUT_DIR = PROJECT_ROOT / "output"


@dataclass
class ModelConfig:
   # --- Data & Target Definition ---
    lookback: int = 6
    target_window: int = 4
    return_threshold: float = 0.025  # 2.5%

   # --- Validation Strategy  ---
    n_folds: int = 5
    test_size: int = 24       # 24 weeks(~6 months) for final test set
    gap: int = 4              # 4 weeks gap between train and test to avoid leakage

    # --- Ensemble Weights ---
    gru_weight: float = 0.4
    logreg_weight: float = 0.6
    random_seed: int = 42

    # --- GRU Hyperparameters ---
    gru_units: int = 12
    gru_batch_size: int = 16
    gru_epochs: int = 50
    gru_patience: int = 5

    # --- Logistic Regression Hyperparameters ---
    logreg_C: float = 1.0
    logreg_penalty: str = "l2"
    logreg_class_weight: str = 'balanced'
    logreg_max_iter: int = 1000

    # --- XGBoost Hyperparameters ---
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 3
    xgb_learning_rate: float = 0.1
    xgb_scale_pos_weight: float = 2.0


@dataclass
class DataConfig:
   raw_filename: str = "mapa_wine_prices_raw.csv"
   price_column_candidates: tuple = ("preciotinto", "price_red")
   campaign_column: str = "campaign"
   week_column: str = "week"


MODEL_CONFIG = ModelConfig()
DATA_CONFIG = DataConfig()
