from typing import Any

import numpy as np
import pandas as pd


def predict_rf(rf_model: Any, X: pd.DataFrame) -> float:
    return float(rf_model.predict(X)[0])


def predict_lstm(lstm_model: Any, X: pd.DataFrame, scaler_x: Any, scaler_y: Any) -> float:
    X_scaled = scaler_x.transform(X)
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
    pred_scaled = lstm_model.predict(X_reshaped, verbose=0)
    return float(scaler_y.inverse_transform(pred_scaled).flatten()[0])
