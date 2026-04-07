import logging
import pickle
from typing import Any

import joblib

from app.infrastructure.artifact_store import ArtifactStore

logger = logging.getLogger(__name__)

_store = ArtifactStore("meat_price_forecast")

TARGET_EXCLUDE_PREFIX = "x"


def load_artifacts() -> tuple[dict[str, Any], dict[str, Any]]:
    bundle_path = _store.path("trained_models_bundle.pkl")

    logger.info("Loading bundle from %s", bundle_path)
    try:
        bundle = joblib.load(bundle_path)
    except Exception:
        logger.warning("joblib.load failed, falling back to pickle.load")
        with bundle_path.open("rb") as f:
            bundle = pickle.load(f)

    if not isinstance(bundle, dict) or "models" not in bundle:
        raise ValueError("Invalid bundle structure: expected dict with 'models' key")

    version = bundle.get("metadata", {}).get("version", "unknown")
    targets = [t for t in bundle["models"] if not t.startswith(TARGET_EXCLUDE_PREFIX)]
    logger.info("Bundle loaded — version=%s, business targets=%s", version, targets)

    lstm_models: dict[str, Any] = {}
    for target, meta in bundle["models"].items():
        if target.startswith(TARGET_EXCLUDE_PREFIX):
            continue
        lstm_data = meta.get("lstm_data")
        if lstm_data and lstm_data.get("config") and lstm_data.get("weights") is not None:
            try:
                from tensorflow.keras import Sequential  # type: ignore[import]
                model = Sequential.from_config(lstm_data["config"])
                model.set_weights(lstm_data["weights"])
                lstm_models[target] = model
                logger.info("LSTM model reconstructed for target '%s'", target)
            except Exception as exc:
                logger.warning("Failed to reconstruct LSTM for target '%s': %s", target, exc)

    return bundle, lstm_models
