from typing import Any

import numpy as np
import pandas as pd


def run_lgbm_predict(model: Any, X: pd.DataFrame) -> float:
    """Run LightGBM inference and return a scalar price in EUR/tonne."""
    result = model.predict(X)
    return float(result[0])
