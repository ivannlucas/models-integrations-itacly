from __future__ import annotations

import pickle
from typing import Any

import pandas as pd

from src.reproducibility.mixed_context import apply_feature_fill_values, validate_feature_columns_for_stage
from src.reproducibility.runtime import official_paths


def predict_trigger(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.Series, pd.Series]:
    paths = official_paths(config)
    with paths["purchase_trigger_artifact"].open("rb") as handle:
        artifact = pickle.load(handle)

    feature_columns = list(artifact["feature_columns"])
    validate_feature_columns_for_stage(feature_columns, stage="trigger")
    fill_values = dict(artifact.get("fill_values", {}))
    x = apply_feature_fill_values(df, feature_columns, fill_values)

    model = artifact["model"]
    flag = pd.Series(model.predict(x), index=df.index, name="purchase_trigger_flag").astype(int)
    if hasattr(model, "predict_proba"):
        proba = pd.Series(model.predict_proba(x)[:, 1], index=df.index, name="purchase_trigger_proba")
    else:
        proba = flag.astype(float).rename("purchase_trigger_proba")
    return proba.clip(lower=0.0, upper=1.0), flag
