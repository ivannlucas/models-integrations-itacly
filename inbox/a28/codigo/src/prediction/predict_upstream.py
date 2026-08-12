from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from src.reproducibility.mixed_context import apply_feature_fill_values
from src.reproducibility.runtime import official_paths


def predict_upstream(df: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    paths = official_paths(config)
    with paths["upstream_predictor_artifact"].open("rb") as handle:
        artifact = pickle.load(handle)
    feature_columns = list(artifact["feature_columns"])
    x = apply_feature_fill_values(df, feature_columns, dict(artifact.get("fill_values", {})))
    prediction = artifact["model"].predict(x)
    return pd.Series(prediction, index=df.index, name="synthetic_procurement_need_pred").clip(lower=0.0)
