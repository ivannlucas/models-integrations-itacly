"""Retraining for ml38 — port of ``python -m scripts.train`` of the AI team.

Source: inbox/a38/codigo/src/main.py::train, src/training/train_models.py::train_family_model and
src/training/train_severity_models.py::train_severity_models (hyperparameters from
config/config.yaml, see constants.TRAIN_*). What is reproduced, step by step:

* Family: GroupKFold(n_splits=min(5, n_groups)) by ``group_id`` — StandardScaler fitted on the
  training fold, RandomForest(800, min_samples_leaf=2, class_weight="balanced",
  random_state=42), accuracy and macro-F1 per fold → mean/std. Then the final scaler + RF on ALL
  rows (that is the served model).
* Severity, per non-healthy family: rows with severity > 0, binarised (1 → 0, >1 → 1); skipped
  if it has < 2 levels or < 2 groups. GroupKFold CV with a fresh MLP per fold, then the final MLP
  on all the family's rows. MLP = Dense(64, relu) → Dropout(0.3) → Dense(32, relu) → Dense(1)
  logit; Adam(lr=0.001), batch 32, 40 epochs, no early stopping. Loss per batch =
  BCE_from_logits + λphys·mean(relu(low_band_norm − sigmoid(logit))) with λphys = 0.3 and
  low_band = raw ``vib_band_low`` min-max normalised with the training rows.

Deliberate, documented differences (behaviour-preserving):

* The original loop uses ``tf.GradientTape``; the server runs Keras with the torch backend (ml1
  sets it), where a tape does not see the gradients. The same loss is written with keras.ops and
  trained with ``model.fit``: the per-sample loss ``bce_i + λ·relu(hb_i − σ(z_i))`` averaged over
  the batch equals ``mean(bce) + λ·mean(penalty)`` exactly; ``fit(shuffle=True)`` reshuffles the
  whole set each epoch, which is what ``Dataset.shuffle(1000)`` does when the family has fewer
  than 1000 rows (all delivered families do; above that the original shuffle is only partial).
* No global seeding (``np.random.seed`` / ``tf.random.set_seed``) and no
  ``keras.backend.clear_session()``: the server is shared. Initialisers and dropout get local
  seeds instead. The severity MLPs were not bit-reproducible in the original either (memoria
  §6.3), only the RandomForest and the folds are.
* The OOF evaluation (``scripts.predict_oof``) is not repeated: it does not produce the served
  model. The CV metrics returned are those of ``scripts.train`` (training_*_metrics.json).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from app.plugins.ml38_wine_machinery_fault_detection.constants import (
    FAMILY_COL,
    FEATURE_NAMES,
    GROUP_COL,
    HEALTHY_FAMILY,
    MIN_TRAIN_GROUPS,
    SEVERITY_COL,
    TRAIN_ACTIVATION,
    TRAIN_BATCH_SIZE,
    TRAIN_DROPOUT,
    TRAIN_EPOCHS,
    TRAIN_HIDDEN_LAYERS,
    TRAIN_LAMBDA_PHYS,
    TRAIN_LEARNING_RATE,
    TRAIN_LOW_BAND_INDEX,
    TRAIN_N_SPLITS,
    TRAIN_RANDOM_STATE,
    TRAIN_RF_CLASS_WEIGHT,
    TRAIN_RF_MIN_SAMPLES_LEAF,
    TRAIN_RF_N_ESTIMATORS,
)
from app.plugins.ml38_wine_machinery_fault_detection.inference import severity_probabilities
from app.plugins.ml38_wine_machinery_fault_detection.model_loader import ModelBundle

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = FEATURE_NAMES + [FAMILY_COL, SEVERITY_COL, GROUP_COL]


# ── data ────────────────────────────────────────────────────────────────────────────────────


def prepare_training_frame(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate the training table and return (X, y_family, y_severity, groups).

    Same layout as data/processed/feature_dataset.parquet. Errors are ValueError (→ 400).
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas requeridas para entrenar: {missing}. Se esperan las 24 "
            f"características más family, severity (entero, -1 si no aplica) y group_id."
        )
    features = df[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce")
    bad = [c for c in FEATURE_NAMES if not np.all(np.isfinite(features[c].to_numpy(float)))]
    if bad:
        raise ValueError(f"Valores vacíos o no numéricos en las características: {bad}.")
    if df[FAMILY_COL].isna().any() or df[GROUP_COL].isna().any():
        raise ValueError("family y group_id no pueden tener valores vacíos.")
    severity = pd.to_numeric(df[SEVERITY_COL], errors="coerce")
    if severity.isna().any() or not np.all(severity == np.round(severity)):
        raise ValueError("severity debe ser un entero en todas las filas (-1 si no aplica).")

    y_family = df[FAMILY_COL].astype(str).to_numpy()
    groups = df[GROUP_COL].to_numpy()
    n_groups = len(np.unique(groups))
    if n_groups < MIN_TRAIN_GROUPS:
        raise ValueError(
            f"Se necesitan al menos {MIN_TRAIN_GROUPS} grupos (group_id) para la validación "
            f"cruzada agrupada; hay {n_groups}."
        )
    if len(np.unique(y_family)) < 2:
        raise ValueError(
            "Se necesitan al menos 2 familias distintas para entrenar el clasificador."
        )
    return features.to_numpy(float), y_family, severity.astype(int).to_numpy(), groups


# ── family (train_models.train_family_model) ──────────────────────────────────────────────────


def _new_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=TRAIN_RF_N_ESTIMATORS,
        min_samples_leaf=TRAIN_RF_MIN_SAMPLES_LEAF,
        class_weight=TRAIN_RF_CLASS_WEIGHT,
        random_state=TRAIN_RANDOM_STATE,
    )


def _cv_summary(scores: list[tuple[float, float]]) -> dict[str, float]:
    """Mean/std over folds of (accuracy, macro-F1) — same keys as training_*_metrics.json."""
    acc, macro = np.array(scores, dtype=float).T
    return {
        "accuracy_mean": float(np.mean(acc)),
        "accuracy_std": float(np.std(acc)),
        "macro_f1_mean": float(np.mean(macro)),
        "macro_f1_std": float(np.std(macro)),
    }


def _scores(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")


def _rf_fold_scores(x, y, train_idx, test_idx) -> tuple[float, float]:
    """Scaler fitted on the training fold, RF, (accuracy, macro-F1) on the test fold."""
    scaler = StandardScaler()
    rf = _new_rf().fit(scaler.fit_transform(x[train_idx]), y[train_idx])
    return _scores(y[test_idx], rf.predict(scaler.transform(x[test_idx])))


def train_family_model(x: np.ndarray, y_family: np.ndarray, groups: np.ndarray):
    """GroupKFold CV metrics + final scaler/RF on every row."""
    n_splits = min(TRAIN_N_SPLITS, len(np.unique(groups)))
    scores = [
        _rf_fold_scores(x, y_family, train_idx, test_idx)
        for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(x, y_family, groups)
    ]
    scaler_final = StandardScaler()
    rf_final = _new_rf().fit(scaler_final.fit_transform(x), y_family)
    return rf_final, scaler_final, _cv_summary(scores)


# ── severity (train_severity_models.train_severity_models) ─────────────────────────────────


def build_severity_model(input_dim: int, seed: int):
    """24 → Dense(64, relu) → Dropout(0.3) → Dense(32, relu) → Dense(1) — _build_severity_model."""
    import keras  # pylint: disable=import-outside-toplevel  # lazy: see model_loader

    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    for i, units in enumerate(TRAIN_HIDDEN_LAYERS):
        x = keras.layers.Dense(
            units, activation=TRAIN_ACTIVATION,
            kernel_initializer=keras.initializers.GlorotUniform(seed=seed + i),
        )(x)
        if i == 0 and TRAIN_DROPOUT > 0:
            x = keras.layers.Dropout(TRAIN_DROPOUT, seed=seed)(x)
    out_init = keras.initializers.GlorotUniform(seed=seed + len(TRAIN_HIDDEN_LAYERS))
    logits = keras.layers.Dense(1, kernel_initializer=out_init)(x)
    return keras.Model(inputs, logits)


def physics_guided_loss(lambda_phys: float = TRAIN_LAMBDA_PHYS):
    """Per-sample loss: bce(y, z) + λ·relu(hb − σ(z)), with ``y_pack = [y, hb]``.

    Keras averages it over the batch ("sum_over_batch_size"), which gives the original
    ``BinaryCrossentropy(from_logits=True)(y, z) + λ·reduce_mean(relu(hb − σ(z)))``.
    """
    from keras import ops  # pylint: disable=import-outside-toplevel

    def loss(y_pack, logits):
        # Tensors of the active backend (torch in the server) even if called with numpy arrays.
        y_pack = ops.convert_to_tensor(y_pack, dtype="float32")
        logits = ops.convert_to_tensor(logits, dtype="float32")
        y = y_pack[:, 0:1]
        hb = y_pack[:, 1]
        bce = ops.mean(ops.binary_crossentropy(y, logits, from_logits=True), axis=-1)
        probs = ops.reshape(ops.sigmoid(logits), (-1,))
        return bce + lambda_phys * ops.relu(hb - probs)

    return loss


def fit_severity_model(x_scaled: np.ndarray, y_binary: np.ndarray, hb_norm: np.ndarray,
                       seed: int):
    """Train one severity MLP and return it uncompiled (picklable, like the delivered ones)."""
    import keras  # pylint: disable=import-outside-toplevel

    model = build_severity_model(x_scaled.shape[1], seed)
    model.compile(optimizer=keras.optimizers.Adam(TRAIN_LEARNING_RATE),
                  loss=physics_guided_loss())
    y_pack = np.stack([y_binary, hb_norm], axis=1).astype(np.float32)
    model.fit(x_scaled.astype(np.float32), y_pack, batch_size=TRAIN_BATCH_SIZE,
              epochs=TRAIN_EPOCHS, shuffle=True, verbose=0)
    clean = build_severity_model(x_scaled.shape[1], seed)
    clean.set_weights(model.get_weights())
    return clean


def _min_max(values: np.ndarray, ref: np.ndarray) -> np.ndarray:
    lo, hi = ref.min(), ref.max()
    return (values - lo) / (hi - lo + 1e-8)


def _mlp_fold_scores(x_fam, y_binary, train_idx, test_idx, seed) -> tuple[float, float]:
    """One severity CV fold: low band normalised and features scaled with the training fold."""
    low_band = x_fam[train_idx, TRAIN_LOW_BAND_INDEX]
    scaler = StandardScaler()
    model = fit_severity_model(scaler.fit_transform(x_fam[train_idx]), y_binary[train_idx],
                               _min_max(low_band, low_band), seed)
    probs = severity_probabilities(model, scaler.transform(x_fam[test_idx]))
    return _scores(y_binary[test_idx], (probs > 0.5).astype(int))


def _family_severity_cv(x_fam, y_binary, groups_fam, n_splits, seed) -> dict[str, float]:
    """GroupKFold CV of the severity MLP of one family (accuracy / macro-F1 per fold)."""
    splits = GroupKFold(n_splits=n_splits).split(x_fam, y_binary, groups=groups_fam)
    return _cv_summary([
        _mlp_fold_scores(x_fam, y_binary, train_idx, test_idx, seed + 100 * (fold + 1))
        for fold, (train_idx, test_idx) in enumerate(splits)
    ])


def _train_family_severity(x_fam, sev, groups_fam, seed):
    """(model, scaler, cv_metrics) of one family, or the reason the original skips it (str)."""
    if len(np.unique(sev)) < 2:
        return "menos de 2 niveles de severidad válidos"
    y_binary = np.where(sev == 1, 0, 1)
    n_splits = min(TRAIN_N_SPLITS, len(np.unique(groups_fam)))
    if n_splits < 2:
        return "menos de 2 grupos con severidad válida"
    metrics = _family_severity_cv(x_fam, y_binary, groups_fam, n_splits, seed)
    low_band = x_fam[:, TRAIN_LOW_BAND_INDEX]
    scaler_final = StandardScaler()
    model = fit_severity_model(scaler_final.fit_transform(x_fam), y_binary,
                               _min_max(low_band, low_band), seed)
    return model, scaler_final, metrics


def train_severity_models(x, y_family, y_severity, groups):
    """Per-family CV metrics + final severity MLPs/scalers (families that can be trained)."""
    models: dict[str, Any] = {}
    scalers: dict[str, Any] = {}
    metrics: dict[str, dict[str, float]] = {}
    skipped: dict[str, str] = {}
    for k, family in enumerate(str(f) for f in np.unique(y_family)):
        if family == HEALTHY_FAMILY:
            continue
        mask = (y_family == family) & (y_severity > 0)
        logger.info("ml38 train: severidad %s (%d filas)", family, int(mask.sum()))
        result = _train_family_severity(x[mask], y_severity[mask], groups[mask],
                                        TRAIN_RANDOM_STATE + 1000 * (k + 1))
        if isinstance(result, str):
            skipped[family] = result
            continue
        models[family], scalers[family], metrics[family] = result
    return models, scalers, metrics, skipped


def train_bundle(df: pd.DataFrame) -> tuple[ModelBundle, dict[str, Any]]:
    """Full ``scripts.train`` on a feature table: returns the new bundle and its CV metrics."""
    x, y_family, y_severity, groups = prepare_training_frame(df)
    rf, scaler, family_metrics = train_family_model(x, y_family, groups)
    sev_models, sev_scalers, sev_metrics, skipped = train_severity_models(
        x, y_family, y_severity, groups
    )
    bundle = ModelBundle(rf, scaler, sev_models, sev_scalers)
    return bundle, {
        "family_cv": family_metrics,
        "severity_cv": sev_metrics,
        "severity_skipped": skipped,
        "n_samples": int(len(x)),
        "n_groups": int(len(np.unique(groups))),
        "families": [str(f) for f in rf.classes_],
    }
