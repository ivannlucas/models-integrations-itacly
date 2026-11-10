"""Retraining logic for ml16 — port of src/training/trainer.py::run_training.

Covers the canonical h=4 pipeline only (the multi-horizon Fase 3 pipeline from
scripts/train_multi_horizon.py is out of scope, see inbox/a16/manifest.yaml known_issues).
Trains fresh XGBoost/LogisticRegression instances from a CSV shaped like
dataset_clasificacion_base.csv (target_animales/target_insumos already computed) — this module
does not reproduce create_targets() nor the raw MAPA+GEE+RASVE ETL, only the modeling stage.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

from app.plugins.ml16_meat_raw_material_price_alert.constants import (
    DEFAULT_LOOKBACK,
    DEFAULT_MANUAL_THRESHOLD_INSUMOS,
    DEFAULT_N_BAGGING,
    DEFAULT_N_SPLITS_CV,
    DEFAULT_SEED,
    DEFAULT_TEST_SIZE,
    DEFAULT_USE_MANUAL_THRESHOLD,
    LOGREG_PARAMS,
    XGBOOST_PARAMS,
)
from app.plugins.ml16_meat_raw_material_price_alert.preprocessing import (
    create_endogenous_features,
    create_sequences,
)

logger = logging.getLogger(__name__)

TARGET_COLS = ["target_animales", "target_insumos"]

INPUT_COLS_COMMON = [
    "indice_animales", "indice_insumos", "mom_indice_animales", "mom_indice_insumos",
    "ma3_indice_animales", "ma3_indice_insumos", "vol3_indice_animales", "vol3_indice_insumos",
    "dev_indice_animales", "dev_indice_insumos", "spread", "month_sin", "month_cos",
]
EXOG_COLS = {
    "target_animales": [
        "animales_afectados", "mom_animales_afectados",
        "animales_afectados_lag1", "animales_afectados_lag2", "animales_afectados_lag3",
    ],
    "target_insumos": [
        "precip_total", "wet_days",
        "precip_total_lag3", "precip_total_lag4", "precip_total_lag5", "precip_total_lag6",
    ],
}


def _build_xgb(spw: float, seed: int) -> xgb.XGBClassifier:
    """Port of trainer.py::build_xgb_animales."""
    return xgb.XGBClassifier(
        **XGBOOST_PARAMS,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=seed,
        verbosity=0,
        n_jobs=1,
    )


def _build_logreg(seed: int) -> LogisticRegression:
    """Port of trainer.py::build_logreg_insumos."""
    return LogisticRegression(class_weight="balanced", random_state=seed, **LOGREG_PARAMS)


def _build_model(target: str, spw: float, seed: int):
    return _build_xgb(spw, seed) if target == "target_animales" else _build_logreg(seed)


def _find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    """Port of trainer.py::find_optimal_threshold.

    Grid search over [0.10, 0.90] step 0.01, max F1.
    """
    best_th, best_f1 = 0.5, -1.0
    for th in np.arange(0.10, 0.91, 0.01):
        preds = (y_proba >= th).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_th, best_f1 = round(float(th), 2), float(f1)
    return best_th, best_f1


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    """Port of trainer.py::evaluate_classification (metrics subset returned by the plugin)."""
    try:
        auc = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": auc,
    }


@dataclass(frozen=True)
class TrainSettings:
    """Hyperparameters of run_training() (config/config.yaml defaults)."""

    seed: int = DEFAULT_SEED
    lookback: int = DEFAULT_LOOKBACK
    test_size: int = DEFAULT_TEST_SIZE
    n_splits_cv: int = DEFAULT_N_SPLITS_CV
    n_bagging: int = DEFAULT_N_BAGGING
    use_manual_threshold: bool = DEFAULT_USE_MANUAL_THRESHOLD
    manual_threshold_insumos: float = DEFAULT_MANUAL_THRESHOLD_INSUMOS


class _Windows(NamedTuple):
    """Per-target scalers and flattened lookback windows, split into train/test."""

    scalers: dict
    x_train_flat: dict
    x_test_flat: dict
    y_train: np.ndarray
    y_test: np.ndarray


def _scale_target(
    df: pd.DataFrame, train_df: pd.DataFrame, cols: list[str], cfg: TrainSettings
) -> tuple[MinMaxScaler, np.ndarray, np.ndarray]:
    """MinMaxScaler fit on train, then flattened windows of the train and test rows."""
    sc = MinMaxScaler()
    sc.fit(train_df[cols])
    scaled = sc.transform(df[cols])
    x_seq = create_sequences(scaled, cfg.lookback)
    x_tr = x_seq[: len(x_seq) - cfg.test_size]
    x_te = x_seq[-cfg.test_size:]
    n_flat = cfg.lookback * len(cols)
    return sc, x_tr.reshape(x_tr.shape[0], n_flat), x_te.reshape(x_te.shape[0], n_flat)


def _build_windows(
    df: pd.DataFrame, input_cols_per_target: dict[str, list[str]], cfg: TrainSettings
) -> _Windows:
    """Scale each target's inputs and build its windows; the last test_size are the test."""
    all_targets = df[TARGET_COLS].values.astype(int)
    train_df = df.iloc[:-cfg.test_size]

    scalers: dict = {}
    x_train_flat: dict = {}
    x_test_flat: dict = {}
    for target, cols in input_cols_per_target.items():
        scalers[target], x_train_flat[target], x_test_flat[target] = _scale_target(
            df, train_df, cols, cfg
        )

    y_seq = all_targets[cfg.lookback:]
    y_train = y_seq[: len(y_seq) - cfg.test_size]
    y_test = y_seq[-cfg.test_size:]
    return _Windows(scalers, x_train_flat, x_test_flat, y_train, y_test)


def _scale_pos_weights(y_train: np.ndarray) -> dict[str, float]:
    """n_neg / n_pos of each target on train (1.0 without positives)."""
    scale_pos_weights = {}
    for i, target in enumerate(TARGET_COLS):
        y_t = y_train[:, i]
        n_neg, n_pos = int((y_t == 0).sum()), int((y_t == 1).sum())
        scale_pos_weights[target] = n_neg / n_pos if n_pos > 0 else 1.0
    return scale_pos_weights


def _oof_thresholds(win: _Windows, spw: dict[str, float], cfg: TrainSettings) -> dict[str, float]:
    """Walk-forward CV (OOF) probabilities on train, then each target's decision threshold."""
    tscv = TimeSeriesSplit(n_splits=cfg.n_splits_cv)
    oof_probas = {t: np.full(len(win.y_train), np.nan) for t in TARGET_COLS}
    oof_true = {t: np.zeros(len(win.y_train)) for t in TARGET_COLS}

    for tr_idx, val_idx in tscv.split(win.y_train):
        for i, target in enumerate(TARGET_COLS):
            x_tr, x_val = win.x_train_flat[target][tr_idx], win.x_train_flat[target][val_idx]
            y_tr, y_val = win.y_train[tr_idx, i], win.y_train[val_idx, i]
            oof_true[target][val_idx] = y_val
            if len(np.unique(y_tr)) < 2:
                oof_probas[target][val_idx] = float(np.unique(y_tr)[0])
                continue
            model = _build_model(target, spw[target], cfg.seed)
            model.fit(x_tr, y_tr)
            oof_probas[target][val_idx] = model.predict_proba(x_val)[:, 1]
    return _select_thresholds(oof_true, oof_probas, cfg)


def _select_thresholds(oof_true: dict, oof_probas: dict, cfg: TrainSettings) -> dict[str, float]:
    """Max-F1 threshold on the OOF predictions (0.5 if one class; manual for insumos)."""
    final_thresholds: dict[str, float] = {}
    for target in TARGET_COLS:
        valid = ~np.isnan(oof_probas[target])
        y_true_oof, y_proba_oof = oof_true[target][valid], oof_probas[target][valid]
        best_th = (
            0.5 if len(np.unique(y_true_oof)) < 2
            else _find_optimal_threshold(y_true_oof, y_proba_oof)[0]
        )
        if target == "target_insumos" and cfg.use_manual_threshold:
            final_thresholds[target] = cfg.manual_threshold_insumos
        else:
            final_thresholds[target] = best_th
    return final_thresholds


def _fit_final(
    win: _Windows, spw: dict[str, float], thresholds: dict[str, float], seed: int
) -> tuple[dict, dict, dict]:
    """Final fit on the full train; returns (models, test probabilities, test predictions)."""
    final_models: dict = {}
    proba_test: dict = {}
    pred_test: dict = {}
    for i, target in enumerate(TARGET_COLS):
        model = _build_model(target, spw[target], seed)
        model.fit(win.x_train_flat[target], win.y_train[:, i])
        final_models[target] = model
        proba = model.predict_proba(win.x_test_flat[target])[:, 1]
        proba_test[target] = proba
        pred_test[target] = (proba >= thresholds[target]).astype(int)
    return final_models, proba_test, pred_test


def _fit_bagging_model(target: str, x_boot: np.ndarray, y_boot: np.ndarray, seed_b: int):
    """One bootstrap model: XGBoost (own scale_pos_weight) for animales, LogReg for insumos."""
    if target == "target_animales":
        n_neg_b = int((y_boot == 0).sum())
        n_pos_b = int((y_boot == 1).sum())
        spw_b = n_neg_b / n_pos_b if n_pos_b > 0 else 1.0
        bmodel = _build_xgb(spw_b, seed_b)
    else:
        bmodel = _build_logreg(seed_b)
    bmodel.fit(x_boot, y_boot)
    return bmodel


def _bootstrap_bagging(win: _Windows, cfg: TrainSettings) -> dict[str, list]:
    """Bootstrap bagging for uncertainty (Fase 5b); degenerate one-class samples are skipped."""
    rng = np.random.RandomState(cfg.seed)  # pylint: disable=no-member  # false positive
    bagging_models: dict[str, list] = {t: [] for t in TARGET_COLS}
    for i, target in enumerate(TARGET_COLS):
        x_tr_t, y_tr_t = win.x_train_flat[target], win.y_train[:, i]
        n_tr = len(y_tr_t)
        for b in range(cfg.n_bagging):
            boot_idx = rng.randint(0, n_tr, size=n_tr)
            x_boot, y_boot = x_tr_t[boot_idx], y_tr_t[boot_idx]
            if len(np.unique(y_boot)) < 2:
                continue  # muestra bootstrap degenerada (una sola clase) -- se omite
            bagging_models[target].append(
                _fit_bagging_model(target, x_boot, y_boot, cfg.seed + b + 1)
            )
    return bagging_models


def train_models(df: pd.DataFrame, *, settings: TrainSettings | None = None) -> dict:
    """Full port of run_training(): feature engineering, walk-forward CV threshold search,
    final fit on train, bootstrap bagging, and hold-out test evaluation.

    Returns a dict with models/scalers/bagging_models/thresholds/metrics/input_cols_per_target/
    n_train/n_test, ready to be persisted by the plugin's train().
    """
    cfg = settings or TrainSettings()
    np.random.seed(cfg.seed)
    input_cols_per_target = {t: INPUT_COLS_COMMON + EXOG_COLS[t] for t in TARGET_COLS}

    df = create_endogenous_features(df)
    df = df.dropna().reset_index(drop=True)
    if len(df) <= cfg.test_size + cfg.lookback:
        raise ValueError(
            f"Tras la ingeniería de variables solo quedan {len(df)} filas útiles: se necesitan "
            f"más de test_size({cfg.test_size}) + lookback({cfg.lookback}) = "
            f"{cfg.test_size + cfg.lookback} filas para poder partir en train/test."
        )

    win = _build_windows(df, input_cols_per_target, cfg)
    spw = _scale_pos_weights(win.y_train)
    final_thresholds = _oof_thresholds(win, spw, cfg)
    final_models, proba_test, pred_test = _fit_final(win, spw, final_thresholds, cfg.seed)
    bagging_models = _bootstrap_bagging(win, cfg)

    # -- Test evaluation ----------------------------------------------------------
    metrics: dict[str, dict[str, float]] = {}
    for i, target in enumerate(TARGET_COLS):
        metrics[target] = _evaluate(win.y_test[:, i], pred_test[target], proba_test[target])
        metrics[target]["threshold"] = final_thresholds[target]

    logger.info(
        "ml16 train_models() done — n_train=%d n_test=%d animales_f1=%.3f insumos_f1=%.3f",
        len(win.y_train), len(win.y_test),
        metrics["target_animales"]["f1"], metrics["target_insumos"]["f1"],
    )
    return {
        "models": final_models,
        "scalers": win.scalers,
        "bagging_models": bagging_models,
        "thresholds": final_thresholds,
        "metrics": metrics,
        "input_cols_per_target": input_cols_per_target,
        "n_train": int(len(win.y_train)),
        "n_test": int(len(win.y_test)),
    }
