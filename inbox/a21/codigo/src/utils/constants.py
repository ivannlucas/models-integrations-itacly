from __future__ import annotations

from pathlib import Path

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
NON_FEATURES: list[str] = []

TRAIN_END = None
SEQ_LEN = 0
MIN_CORR = 0.0
HORIZONS: list[int] = []

DRNN_HIDDEN = 0
DRNN_LAYERS = 0
DRNN_DROPOUT = 0.0
DRNN_EPOCHS = 0
DRNN_LR = 0.0

CEREAL_SUP_MAP: dict[str, str] = {}
