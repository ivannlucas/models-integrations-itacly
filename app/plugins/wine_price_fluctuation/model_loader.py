import json
import logging

import joblib

from app.infrastructure.artifact_store import ArtifactStore

logger = logging.getLogger(__name__)

_store = ArtifactStore("wine_price_fluctuation")


def load_artifacts() -> tuple:
    ml_model_path = _store.path("ml_model.pkl")
    scaler_path = _store.path("scaler.pkl")
    feature_schema_path = _store.path("feature_schema.json")
    model_config_path = _store.path("model_config.json")

    logger.info("Loading ml_model from %s", ml_model_path)
    ml_model = joblib.load(ml_model_path)

    logger.info("Loading scaler from %s", scaler_path)
    scaler = joblib.load(scaler_path)

    logger.info("Loading feature schema from %s", feature_schema_path)
    with open(feature_schema_path) as f:
        schema = json.load(f)
    feature_columns: list[str] = schema["feature_columns"]

    logger.info("Loading model config from %s", model_config_path)
    with open(model_config_path) as f:
        model_cfg = json.load(f)
    model_type: str = model_cfg["model_type"]

    logger.info("All artifacts loaded — model_type=%s, features=%s", model_type, feature_columns)
    return ml_model, scaler, feature_columns, model_type
