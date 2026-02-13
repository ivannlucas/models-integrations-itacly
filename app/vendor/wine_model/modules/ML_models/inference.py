"""
Inference utilities for tabular wine price forecasting models.

This module loads the trained production artifacts and exposes
high-level prediction functions:
- Evaluate the final model on the last N weeks of the historical
  dataset to obtain test metrics.
- Run predictions on new CSV files with the same schema as the
  training data, applying the same ETL and feature pipeline.

All predictions use a fixed horizon of 4 weeks, consistent with the
training target definition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional

import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)

from modules.common.config import (
    MODEL_CONFIG,
    DATA_CONFIG,
    RAW_DATA_DIR,
    MODELS_PROD_DIR,
)
from modules.common.data import load_raw_data, clean_and_index_data
from modules.common.features import generate_technical_features, transform_features


def _load_scaler_and_schema() -> Tuple[object, Dict]:
    """Load the fitted scaler and feature schema from the production directory."""
    scaler_path = MODELS_PROD_DIR / "scaler.pkl"
    schema_path = MODELS_PROD_DIR / "feature_schema.json"

    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    if not schema_path.exists():
        raise FileNotFoundError(f"Feature schema not found at {schema_path}")

    scaler = joblib.load(scaler_path)
    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)
    return scaler, schema


def _load_model_config() -> Dict:
    """Load the model configuration (model_type, etc.)."""
    config_path = MODELS_PROD_DIR / "model_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found at {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_models(model_type: str) -> object:
    """
    Load the production model according to model_type.

    Returns:
        The loaded ML model (Logistic Regression or XGBoost).
    """
    ml_model_path = MODELS_PROD_DIR / "ml_model.pkl"
    if not ml_model_path.exists():
        raise FileNotFoundError(f"ML model not found at {ml_model_path}")

    ml_model = joblib.load(ml_model_path)
    return ml_model


def _compute_test_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute metrics on the test set."""
    if len(np.unique(y_true)) == 1:
        auc = 0.5
    else:
        auc = roc_auc_score(y_true, y_prob)

    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred)

    return {
        "auc": float(auc),
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
    }


def evaluate_on_last_weeks(
    raw_path: Optional[Path] = None,
    test_weeks: int = MODEL_CONFIG.test_size,
) -> Dict:
    """
    Evaluate production models on the last N weeks of available data.

    This function automatically filters out recent weeks where the target
    cannot be calculated (because the future window has not closed yet),
    ensuring metrics are computed only on verifiable ground truth.

    Args:
        raw_path: Path to raw data CSV. If None, uses default from config.
        test_weeks: Number of last weeks to attempt to use as test set.

    Returns:
        Dictionary with model type, test weeks, sample count, metrics, and predictions_df.
    """
    # 1. Load and clean raw data
    if raw_path is None:
        raw_path = RAW_DATA_DIR / DATA_CONFIG.raw_filename

    df_raw = load_raw_data(raw_path)
    df_price = clean_and_index_data(df_raw)

    # 2. Build features on FULL dataset
    df_features = generate_technical_features(
        df_price,
        target_window=MODEL_CONFIG.target_window,
        return_threshold=MODEL_CONFIG.return_threshold,
    )

    # 3. Load artifacts
    scaler, schema = _load_scaler_and_schema()
    model_cfg = _load_model_config()
    model_type = model_cfg.get("model_type", "logreg")
    ml_model = _load_models(model_type=model_type)

    feature_cols = schema["feature_columns"]
    X_full = df_features[feature_cols]

    # Scaling the full dataset to ensure continuity
    X_scaled = transform_features(scaler, X_full)

    # 4. Define Valid Test Set (Filter out NaNs from open future windows)
    df_test_candidates = df_features.iloc[-test_weeks:].copy()

    # Drop rows where target is NaN (future prediction window hasn't closed yet)
    df_test_valid = df_test_candidates.dropna(subset=["target"])

    if df_test_valid.empty:
        raise ValueError(
            "No valid test data found. The requested test_weeks window might "
            "be entirely within the 'blind' future period (target is NaN)."
        )

    # Get ground truth and dates for the valid subset
    y_true_test = df_test_valid["target"].values.astype(int)
    test_dates = df_test_valid.index
    n_valid_samples = len(df_test_valid)

    # Calculate slicing indices for X arrays
    start_idx = -test_weeks
    end_idx = start_idx + n_valid_samples

    # If end_idx is 0, it means we want up to the end (no NaNs found).
    slice_obj = slice(start_idx, None) if end_idx == 0 else slice(start_idx, end_idx)

    # Slice tabular data
    X_test_scaled = X_scaled[slice_obj]

    # Predict
    if model_type in {"logreg", "xgboost"}:
        y_prob_test = ml_model.predict_proba(X_test_scaled)[:, 1]
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Validation check
    if y_prob_test is None or len(y_prob_test) != len(y_true_test):
        raise ValueError(
            f"Prediction mismatch. True labels: {len(y_true_test)}, "
            f"Predictions: {len(y_prob_test) if y_prob_test is not None else 0}"
        )

    # Compute metrics
    metrics = _compute_test_metrics(y_true=y_true_test, y_prob=y_prob_test)

    # Create predictions DataFrame
    y_pred_test = (y_prob_test >= 0.5).astype(int)
    predictions_df = pd.DataFrame(
        {
            "date": test_dates,
            "y_true": y_true_test,
            "y_prob": y_prob_test,
            "y_pred": y_pred_test,
        }
    ).set_index("date")

    return {
        "model_type": model_type,
        "test_weeks": test_weeks,
        "n_test_samples": n_valid_samples,
        "metrics": metrics,
        "predictions_df": predictions_df,
    }


def predict_from_csv(
    input_path: Path,
) -> pd.DataFrame:
    """
    Run inference on a new CSV file with the same schema as the training data.

    Steps:
    - Load raw CSV and run ETL (campaign/week -> date, price cleaning).
    - Build technical features (same as in training), but ignore rows with
      missing target (we only care about future predictions).
    - Apply the production scaler and feature schema.
    - Use the production model (logreg / xgboost) to compute
      probabilities of "price goes up over next 4 weeks".
    - Return a DataFrame with dates, features and predicted probabilities.

    Args:
        input_path: Path to the new CSV file.

    Returns:
        DataFrame with index as date and columns:
            - all feature columns used
            - "pred_proba_up" with predicted probability of class 1.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    # 1. Load and clean
    df_raw = pd.read_csv(input_path)
    df_price = clean_and_index_data(df_raw)

    # 2. Build features (target is created but not required for prediction)
    df_features = generate_technical_features(
        df_price,
        target_window=MODEL_CONFIG.target_window,
        return_threshold=MODEL_CONFIG.return_threshold,
    )

    # 3. Load artifacts
    scaler, schema = _load_scaler_and_schema()
    model_cfg = _load_model_config()
    model_type = model_cfg.get("model_type", "logreg")
    ml_model = _load_models(model_type=model_type)

    feature_cols = schema["feature_columns"]
    X_df = df_features[feature_cols]

    # 4. Transform features
    X_scaled = transform_features(scaler, X_df)

    # 5. Predict probabilities
    if model_type in {"logreg", "xgboost"}:
        y_prob = ml_model.predict_proba(X_scaled)[:, 1]
    else:
        raise ValueError(f"Unknown model_type in config: {model_type}")

    # 6. Build output DataFrame
    df_out = df_features.copy()
    df_out["pred_proba_up"] = y_prob

    return df_out
