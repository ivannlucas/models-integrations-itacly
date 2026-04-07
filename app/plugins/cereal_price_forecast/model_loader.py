import json
import logging
from typing import Any

import joblib

from app.infrastructure.artifact_store import ArtifactStore

logger = logging.getLogger(__name__)

_store = ArtifactStore("cereal_price_forecast")

PRODUCT_TO_FILE: dict[str, str] = {
    "Durum wheat": "lgbm_durum_wheat.pkl",
    "Milling wheat": "lgbm_milling_wheat.pkl",
    "Feed barley": "lgbm_feed_barley.pkl",
    "Malting barley": "lgbm_malting_barley.pkl",
    "Feed maize": "lgbm_feed_maize.pkl",
}


def load_artifacts() -> tuple[dict[str, Any], list[str], dict[str, float]]:
    feature_list_path = _store.path("feature_list.txt")
    with open(feature_list_path) as f:
        feature_cols = [line.strip() for line in f if line.strip()]
    logger.info("Loaded %d features from %s", len(feature_cols), feature_list_path)

    models: dict[str, Any] = {}
    for product, filename in PRODUCT_TO_FILE.items():
        model_path = _store.path(filename)
        logger.info("Loading model for '%s' from %s", product, model_path)
        models[product] = joblib.load(model_path)

    feature_medians: dict[str, float] = {}
    try:
        medians_path = _store.path("feature_medians.json")
        with open(medians_path) as f:
            feature_medians = json.load(f)
        logger.info("Loaded feature medians (%d entries)", len(feature_medians))
    except FileNotFoundError:
        logger.warning("feature_medians.json not found — median imputation disabled")

    return models, feature_cols, feature_medians
