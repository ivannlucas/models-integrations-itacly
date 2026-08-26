from pathlib import Path

from src.utils.logging import get_logger

try:
    from src.utils.constants import (
        PROJECT, DATA, RAW, PROCESSED, SPLITS, UTILS_DIR, MODELS_DIR,
        DATASET_MENSUAL_CSV, DATASET_FE_CSV,
        TARGET_REG, TARGET_CLF, NON_FEATURES,
        TRAIN_END, SEQ_LEN, MIN_CORR, HORIZONS,
        DRNN_HIDDEN, DRNN_LAYERS, DRNN_DROPOUT, DRNN_EPOCHS, DRNN_LR,
        CEREAL_SUP_MAP,
    )
except Exception:
    PROJECT = Path(".")
    DATA = PROJECT / "data"
    RAW = DATA / "raw"
    PROCESSED = DATA / "processed"
    SPLITS = DATA / "splits"
    UTILS_DIR = PROCESSED / "auto" / "utils"
    MODELS_DIR = PROJECT / "models"
    DATASET_MENSUAL_CSV = PROCESSED / "dataset_v7_mensual.csv"
    DATASET_FE_CSV = PROCESSED / "dataset_fe.csv"
    TARGET_REG = "target"
    TARGET_CLF = "target_clf"
    NON_FEATURES = []
    TRAIN_END = None
    SEQ_LEN = 0
    MIN_CORR = 0.0
    HORIZONS = []
    DRNN_HIDDEN = 0
    DRNN_LAYERS = 0
    DRNN_DROPOUT = 0.0
    DRNN_EPOCHS = 0
    DRNN_LR = 0.0
    CEREAL_SUP_MAP = {}
from src.utils.io import (
    load_dataset, save_dataset,
    save_pickle, load_pickle,
    save_json, load_json,
    save_splits, load_splits,
)
