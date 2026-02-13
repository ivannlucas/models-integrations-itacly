"""
Sequential feature utilities for GRU models.

This module provides the create_sequences function to convert tabular
data into sequences for time-series models like GRU.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert tabular data into sequences for time-series models (e.g., GRU).

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Target array of shape (n_samples,).
        lookback: Number of past time steps to include in each sequence.

    Returns:
        (X_seq, y_seq):
            X_seq: shape (n_sequences, lookback, n_features)
            y_seq: shape (n_sequences,) aligned with the last step of each sequence
    """
    n_samples = X.shape[0]
    if n_samples < lookback:
        raise ValueError(
            f"Not enough samples ({n_samples}) to create sequences with lookback={lookback}."
        )

    X_seq_list = []
    y_seq_list = []

    for i in range(n_samples - lookback + 1):
        X_seq_list.append(X[i : i + lookback])
        # Target aligned with the last step of the sequence
        y_seq_list.append(y[i + lookback - 1])

    X_seq = np.array(X_seq_list, dtype=np.float32)
    y_seq = np.array(y_seq_list, dtype=np.float32)

    return X_seq, y_seq
