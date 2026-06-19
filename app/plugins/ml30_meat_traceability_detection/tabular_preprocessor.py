"""Vendored ``TabularPreprocessor`` from the modelo30 training repo.

``preprocessor.pkl`` is a pickled instance of this class under the module path
``src.training.preprocessing``. ``model_loader`` aliases that path to this module
in ``sys.modules`` so joblib.load resolves the class. Kept verbatim from the
source so transforms are byte-identical to training.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


class TabularPreprocessor:
    """Median-fill + standardize numeric features and one-hot encode categoricals."""

    def __init__(self, numeric_features: List[str], categorical_features: List[str]) -> None:
        """Store the numeric/categorical feature contract; stats are filled by fit()."""
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features)
        self.numeric_medians: Dict[str, float] = {}
        self.numeric_means: Dict[str, float] = {}
        self.numeric_stds: Dict[str, float] = {}
        self.categorical_vocab: Dict[str, List[str]] = {}
        self.feature_names_: List[str] = []
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        """Fit medians/means/stds and categorical vocabularies from a DataFrame."""
        for col in self.numeric_features:
            series = pd.to_numeric(df[col], errors="coerce")
            median = float(series.median()) if pd.notna(series.median()) else 0.0
            filled = series.fillna(median)
            mean = float(filled.mean()) if pd.notna(filled.mean()) else 0.0
            std = float(filled.std()) if pd.notna(filled.std()) else 0.0
            if std <= 1e-9:
                std = 1.0
            self.numeric_medians[col] = median
            self.numeric_means[col] = mean
            self.numeric_stds[col] = std

        for col in self.categorical_features:
            values = df[col].astype("string").fillna("__MISSING__")
            vocab = sorted(values.unique().tolist())
            if "__MISSING__" not in vocab:
                vocab.append("__MISSING__")
            self.categorical_vocab[col] = vocab

        self.feature_names_ = self.numeric_features.copy()
        for col in self.categorical_features:
            self.feature_names_.extend([f"{col}__{cat}" for cat in self.categorical_vocab[col]])
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform a DataFrame into the standardized + one-hot feature matrix."""
        if not self.fitted:
            raise RuntimeError("TabularPreprocessor no esta ajustado.")

        numeric_matrix = []
        for col in self.numeric_features:
            series = pd.to_numeric(df[col], errors="coerce").fillna(self.numeric_medians[col])
            norm = (series.to_numpy(dtype=float) - self.numeric_means[col]) / self.numeric_stds[col]
            numeric_matrix.append(norm.reshape(-1, 1))
        x_num = np.hstack(numeric_matrix) if numeric_matrix else np.zeros((len(df), 0), dtype=float)

        cat_blocks = []
        for col in self.categorical_features:
            vocab = self.categorical_vocab[col]
            value_to_idx = {v: i for i, v in enumerate(vocab)}
            values = df[col].astype("string").fillna("__MISSING__").to_numpy()
            block = np.zeros((len(df), len(vocab)), dtype=float)
            for i, value in enumerate(values):
                idx = value_to_idx.get(str(value), value_to_idx["__MISSING__"])
                block[i, idx] = 1.0
            cat_blocks.append(block)
        x_cat = np.hstack(cat_blocks) if cat_blocks else np.zeros((len(df), 0), dtype=float)

        return np.hstack([x_num, x_cat]).astype(float)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit then transform in one call."""
        return self.fit(df).transform(df)
