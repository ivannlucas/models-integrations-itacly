"""Keep a precomputed SHAP background consistent with a retrained model's input scaler.

The DNSL plugins (ml43, ml45) ship ``xai_background.npy``: reference windows for SHAP's
GradientExplainer, already scaled with the base model's ``scaler_x``. In the original
repositories that file is rebuilt from ``data/raw/xai_background.csv`` with the *current*
``scaler_x`` every time the data is reprocessed, so a retrain always gets a background in its
own input space. The plugins only have the scaled ``.npy``, so on retrain they must re-express
it in the new scaler's space — otherwise SHAP compares the new model's inputs against
reference windows on a different scale and every temporal attribution (and everything derived
from it: dominant variables, subsystem scores, interpretive state) is distorted.
"""
from __future__ import annotations

import numpy as np


def rescale_background(background: np.ndarray | None, fitted_scaler, new_scaler) -> np.ndarray | None:
    """Re-express ``background`` [N, T, F], scaled with ``fitted_scaler``, in ``new_scaler``'s space.

    Both scalers are per-feature affine transforms fitted on rows of F features (sklearn
    ``StandardScaler`` and the like), so undoing the first and applying the second gives the
    same windows the original pipeline would get by rescaling the raw background CSV. Returns
    ``background`` untouched when there is nothing to convert (no background, or a missing
    scaler on either side).
    """
    if background is None or fitted_scaler is None or new_scaler is None:
        return background
    windows = np.asarray(background)
    n_features = windows.shape[-1]
    raw = fitted_scaler.inverse_transform(windows.reshape(-1, n_features))
    return new_scaler.transform(raw).reshape(windows.shape).astype(windows.dtype, copy=False)
