"""«desconocido» gate on a network's own features: inputs unlike the ones it was trained on.

A classifier without a "none of these" class spreads 100 % of its softmax over its known classes,
so on inputs from another domain it still answers one of them with high confidence and a
confidence threshold cannot catch them. This gate scores the network's pooled features instead:

* ``knn``: cosine distance to the nearest training input (deep k-NN, Sun et al., ICML 2022);
* ``logreg``: a logistic regression (outlier exposure, Hendrycks et al., ICLR 2019) trained on
  those features + the k-NN distance, real training inputs vs inputs of other domains. It
  generalises to unseen domains better than k-NN alone at the same share of real inputs kept
  (outputs/a08/README_ood.md, outputs/a10/README_ood.md).

The threshold keeps a fixed share (``keep``, 95 % by default) of real inputs that were not used
to fit the gate. The plugin decides what to do above it (ml8 per image, ml10 per detection) and
never changes the classifier itself. Model-specific code (which features, which output fields)
lives in each plugin's ``ood.py``.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

OOD_REFERENCE_FILENAME = "ood_reference.npz"
UNKNOWN_LABEL = "desconocido"
# Share of held-out real inputs that must stay below the threshold (not «desconocido»).
DEFAULT_KEEP = 0.95
# The k-NN distance (0..2) is scaled before joining the ~unit-norm features in the regression.
_KNN_SCALE = 10.0
_LOGREG_C = 0.1


def _l2n(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x).astype(np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


@dataclass(frozen=True)
class OodReference:
    """Training-input bank, optional logistic regression and decision threshold."""

    bank: np.ndarray  # (N, D) float32, rows L2-normalised
    threshold: float
    keep: float
    # Logistic regression on [l2n(features), KNN_SCALE * knn]; None = k-NN distance only.
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    coef: np.ndarray | None = None
    intercept: float = 0.0

    @property
    def method(self) -> str:
        """``logreg`` (outlier exposure + k-NN) or ``knn`` (no outlier inputs available)."""
        return "knn" if self.coef is None else "logreg"

    def knn(self, features: np.ndarray, exclude_self: bool = False) -> np.ndarray:
        """1 - cosine similarity to the nearest bank input (0..2)."""
        sims = _l2n(features) @ self.bank.T
        if exclude_self:  # scoring the bank itself: skip each row's own match
            np.fill_diagonal(sims, -1.0)
        return 1.0 - sims.max(axis=1)

    def design(self, features: np.ndarray, exclude_self: bool = False) -> np.ndarray:
        """Regression input: L2-normalised features + scaled k-NN distance."""
        return np.c_[_l2n(features), _KNN_SCALE * self.knn(features, exclude_self)]

    def score(self, features: np.ndarray) -> np.ndarray:
        """OOD score per row (higher = less like the training inputs)."""
        if self.coef is None:
            return self.knn(features)
        z = (self.design(features) - self.mean) / self.scale
        return 1.0 / (1.0 + np.exp(-(z @ self.coef + self.intercept)))

    def with_threshold(self, held_out_scores: np.ndarray) -> "OodReference":
        """Copy whose threshold keeps ``keep`` of these held-out real inputs' scores."""
        threshold = float(np.quantile(held_out_scores, self.keep))
        return OodReference(bank=self.bank, threshold=threshold, keep=self.keep, mean=self.mean,
                            scale=self.scale, coef=self.coef, intercept=self.intercept)

    def save(self, path: str) -> None:
        """Write the reference as a compressed .npz next to the checkpoint."""
        extra = {} if self.coef is None else {
            "mean": self.mean, "scale": self.scale, "coef": self.coef, "intercept": self.intercept}
        np.savez_compressed(path, bank=self.bank.astype(np.float16), threshold=self.threshold,
                            keep=self.keep, **extra)


def load_reference(folder: str, model: str = "") -> OodReference | None:
    """The OodReference stored in *folder*, or None (gate disabled) if it is not there.

    *model* only prefixes the warning logged when the file is missing.
    """
    path = os.path.join(folder, OOD_REFERENCE_FILENAME)
    if not os.path.exists(path):
        logger.warning("%s: %s no encontrado en %s — sin detección de «desconocido»",
                       model or "ood", OOD_REFERENCE_FILENAME, folder)
        return None
    with np.load(path) as data:
        lr = {k: np.asarray(data[k], dtype=np.float64)
              for k in ("mean", "scale", "coef") if k in data}
        return OodReference(bank=_l2n(data["bank"]), threshold=float(data["threshold"]),
                            keep=float(data["keep"]),
                            intercept=float(data["intercept"]) if "intercept" in data else 0.0,
                            **lr)


def fit_reference(bank_features: np.ndarray, outlier_features: np.ndarray | None = None,
                  keep: float = DEFAULT_KEEP) -> OodReference:
    """Fit the gate on training-input features (+ outlier inputs'); threshold still unset.

    With outliers it fits the logistic regression (bank rows scored leave-one-out, so each one
    does not match itself); without them the score is the k-NN distance.
    """
    if len(bank_features) < 2:
        raise ValueError("Datos insuficientes para calibrar la detección de «desconocido».")
    ref = OodReference(bank=_l2n(bank_features), threshold=float("inf"), keep=keep)
    if outlier_features is None or len(outlier_features) == 0:
        return ref
    # pylint: disable-next=import-outside-toplevel  # sklearn only when fitting (train/script)
    from sklearn.linear_model import LogisticRegression

    x = np.r_[ref.design(bank_features, exclude_self=True), ref.design(outlier_features)]
    y = np.r_[np.zeros(len(bank_features)), np.ones(len(outlier_features))]
    mean, scale = x.mean(axis=0), x.std(axis=0) + 1e-6
    clf = LogisticRegression(C=_LOGREG_C, max_iter=5000, class_weight="balanced")
    clf.fit((x - mean) / scale, y)
    return OodReference(bank=ref.bank, threshold=float("inf"), keep=keep, mean=mean,
                        scale=scale, coef=clf.coef_[0].astype(np.float64),
                        intercept=float(clf.intercept_[0]))


def build_reference(bank_features: np.ndarray, held_out_features: np.ndarray,
                    outlier_features: np.ndarray | None = None,
                    keep: float = DEFAULT_KEEP) -> OodReference:
    """Fit on the bank (+ outliers) and set the threshold on held-out real inputs.

    The held-out inputs must not be in the bank: use the validation split, never training.
    """
    if len(held_out_features) == 0:
        raise ValueError("Sin datos de validación para calibrar la detección de «desconocido».")
    ref = fit_reference(bank_features, outlier_features, keep)
    return ref.with_threshold(ref.score(held_out_features))
