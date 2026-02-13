"""
Model definitions for tabular wine price forecasting (production).

This module defines the predictive models used in production:
- A tabular Logistic Regression classifier for engineered features.
- XGBoost classifier for tabular features.
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from modules.common.config import MODEL_CONFIG


def build_logistic_model() -> LogisticRegression:
    """
    Build a Logistic Regression classifier for tabular features.

    Uses a simple but robust configuration with class balancing and L2 penalty.
    """
    model = LogisticRegression(
        C=MODEL_CONFIG.logreg_C,
        penalty=MODEL_CONFIG.logreg_penalty,
        class_weight=MODEL_CONFIG.logreg_class_weight,
        max_iter=MODEL_CONFIG.logreg_max_iter,
        random_state=MODEL_CONFIG.random_seed,
    )
    return model


def build_xgboost_model() -> XGBClassifier:
    """
    Build an XGBoost classifier for tabular features.

    Uses default parameters with class balancing.
    """
    model = XGBClassifier(
        random_state=MODEL_CONFIG.random_seed,
        n_estimators=MODEL_CONFIG.xgb_n_estimators,
        max_depth=MODEL_CONFIG.xgb_max_depth,
        learning_rate=MODEL_CONFIG.xgb_learning_rate,
        scale_pos_weight=MODEL_CONFIG.xgb_scale_pos_weight,  # Adjust for class imbalance if needed
    )
    return model
