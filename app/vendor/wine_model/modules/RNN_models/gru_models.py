"""
GRU model definitions for experimental wine price forecasting.

This module defines:
- A small GRU-based neural network for sequential inputs.
- A simple ensemble wrapper that combines GRU and LogisticRegression
  probabilities with configurable weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization, GaussianNoise, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

from modules.common.config import MODEL_CONFIG


@dataclass
class EnsembleWeights:
    """Weights for combining GRU and LogisticRegression outputs."""
    gru_weight: float = MODEL_CONFIG.gru_weight
    logreg_weight: float = MODEL_CONFIG.logreg_weight

    def normalize(self) -> Tuple[float, float]:
        total = self.gru_weight + self.logreg_weight
        if total <= 0:
            raise ValueError("Ensemble weights must be positive.")
        return self.gru_weight / total, self.logreg_weight / total


def build_gru_model(input_shape: Tuple[int, int]) -> Model:
    """
    Build a small GRU-based Keras model for sequential inputs.

    Architecture:
    - Optional GaussianNoise for robustness.
    - Single GRU layer with strong L2 regularization.
    - BatchNormalization + Dropout.
    - Final Dense(1, sigmoid) for binary classification.
    """
    lookback, n_features = input_shape

    inputs = Input(shape=(lookback, n_features))
    x = GaussianNoise(0.01)(inputs)

    x = GRU(
        units=MODEL_CONFIG.gru_units,
        return_sequences=False,
        kernel_regularizer=regularizers.l2(0.05),
        recurrent_regularizer=regularizers.l2(0.05),
    )(x)

    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


class GRULogisticEnsemble:
    """
    Simple ensemble of GRU and Logistic Regression.

    The ensemble combines the predicted probabilities from both models
    using weighted averaging. Weights are normalized to sum to 1.
    """

    def __init__(
        self,
        gru_model: Model,
        logreg_model: LogisticRegression,
        weights: EnsembleWeights | None = None,
    ) -> None:
        if weights is None:
            weights = EnsembleWeights()
        self.gru_model = gru_model
        self.logreg_model = logreg_model
        self.gru_weight, self.logreg_weight = weights.normalize()

    def predict_proba(
        self,
        X_tabular: np.ndarray,
        X_seq: np.ndarray,
    ) -> np.ndarray:
        """
        Predict ensemble probabilities for class 1.

        Args:
            X_tabular: 2D array (n_samples, n_features) for Logistic Regression.
            X_seq: 3D array (n_sequences, lookback, n_features) for GRU.

        Returns:
            1D array of ensemble probabilities in [0, 1].

        Notes:
            The GRU sequences and tabular features must be aligned so that
            they refer to the same time steps. This alignment is handled
            in the training/inference pipeline, not here.
        """
        # LogisticRegression: predict_proba returns (n_samples, 2)
        p_logreg = self.logreg_model.predict_proba(X_tabular)[:, 1]

        # GRU: predict returns (n_samples, 1)
        p_gru = self.gru_model.predict(X_seq, verbose=0).reshape(-1)

        if p_logreg.shape[0] != p_gru.shape[0]:
            raise ValueError(
                f"Logistic and GRU predictions have different lengths: "
                f"{p_logreg.shape[0]} vs {p_gru.shape[0]}"
            )

        p_ensemble = (
            self.gru_weight * p_gru + self.logreg_weight * p_logreg
        )
        # Clip to [0, 1] for numerical safety
        return np.clip(p_ensemble, 0.0, 1.0)
