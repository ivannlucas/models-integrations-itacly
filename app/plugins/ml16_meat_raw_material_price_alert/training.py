"""Retraining logic for ml16 — port of src/training/trainer.py::run_training.

Covers the canonical h=4 pipeline only (the multi-horizon Fase 3 pipeline from
scripts/train_multi_horizon.py is out of scope, see inbox/a16/manifest.yaml known_issues).
Trains fresh XGBoost/LogisticRegression instances from a CSV shaped like
dataset_clasificacion_base.csv (target_animales/target_insumos already computed) — this module
does not reproduce create_targets() nor the raw MAPA+GEE+RASVE ETL, only the modeling stage.
"""
from __future__ import annotations

import logging

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
    """Port of trainer.py::find_optimal_threshold — grid search over [0.10, 0.90] step 0.01, max F1."""
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


def train_models(
    df: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
    lookback: int = DEFAULT_LOOKBACK,
    test_size: int = DEFAULT_TEST_SIZE,
    n_splits_cv: int = DEFAULT_N_SPLITS_CV,
    n_bagging: int = DEFAULT_N_BAGGING,
    use_manual_threshold: bool = DEFAULT_USE_MANUAL_THRESHOLD,
    manual_threshold_insumos: float = DEFAULT_MANUAL_THRESHOLD_INSUMOS,
) -> dict:
    """Full port of run_training(): feature engineering, walk-forward CV threshold search,
    final fit on train, bootstrap bagging, and hold-out test evaluation.

    Returns a dict with models/scalers/bagging_models/thresholds/metrics/input_cols_per_target/
    n_train/n_test, ready to be persisted by the plugin's train().
    """
    np.random.seed(seed)
    input_cols_per_target = {t: INPUT_COLS_COMMON + EXOG_COLS[t] for t in TARGET_COLS}

    df = create_endogenous_features(df)
    df = df.dropna().reset_index(drop=True)
    if len(df) <= test_size + lookback:
        raise ValueError(
            f"Tras la ingeniería de variables solo quedan {len(df)} filas útiles: se necesitan "
            f"más de test_size({test_size}) + lookback({lookback}) = {test_size + lookback} "
            "filas para poder partir en train/test."
        )

    all_targets = df[TARGET_COLS].values.astype(int)
    train_df = df.iloc[:-test_size]

    scalers: dict = {}
    x_train_flat: dict = {}
    x_test_flat: dict = {}
    for target, cols in input_cols_per_target.items():
        sc = MinMaxScaler()
        sc.fit(train_df[cols])
        scaled = sc.transform(df[cols])
        x_seq = create_sequences(scaled, lookback)
        x_tr = x_seq[: len(x_seq) - test_size]
        x_te = x_seq[-test_size:]
        n_flat = lookback * len(cols)
        scalers[target] = sc
        x_train_flat[target] = x_tr.reshape(x_tr.shape[0], n_flat)
        x_test_flat[target] = x_te.reshape(x_te.shape[0], n_flat)

    y_seq = all_targets[lookback:]
    y_train = y_seq[: len(y_seq) - test_size]
    y_test = y_seq[-test_size:]

    scale_pos_weights = {}
    for i, target in enumerate(TARGET_COLS):
        y_t = y_train[:, i]
        n_neg, n_pos = int((y_t == 0).sum()), int((y_t == 1).sum())
        scale_pos_weights[target] = n_neg / n_pos if n_pos > 0 else 1.0

    # -- Walk-forward CV (OOF) -------------------------------------------------
    tscv = TimeSeriesSplit(n_splits=n_splits_cv)
    oof_probas = {t: np.full(len(y_train), np.nan) for t in TARGET_COLS}
    oof_true = {t: np.zeros(len(y_train)) for t in TARGET_COLS}

    for tr_idx, val_idx in tscv.split(y_train):
        for i, target in enumerate(TARGET_COLS):
            x_tr, x_val = x_train_flat[target][tr_idx], x_train_flat[target][val_idx]
            y_tr, y_val = y_train[tr_idx, i], y_train[val_idx, i]
            oof_true[target][val_idx] = y_val
            if len(np.unique(y_tr)) < 2:
                oof_probas[target][val_idx] = float(np.unique(y_tr)[0])
                continue
            model = _build_model(target, scale_pos_weights[target], seed)
            model.fit(x_tr, y_tr)
            oof_probas[target][val_idx] = model.predict_proba(x_val)[:, 1]

    final_thresholds: dict[str, float] = {}
    for target in TARGET_COLS:
        valid = ~np.isnan(oof_probas[target])
        y_true_oof, y_proba_oof = oof_true[target][valid], oof_probas[target][valid]
        best_th = 0.5 if len(np.unique(y_true_oof)) < 2 else _find_optimal_threshold(y_true_oof, y_proba_oof)[0]
        if target == "target_insumos" and use_manual_threshold:
            final_thresholds[target] = manual_threshold_insumos
        else:
            final_thresholds[target] = best_th

    # -- Final fit on full train ------------------------------------------------
    final_models: dict = {}
    proba_test: dict = {}
    pred_test: dict = {}
    for i, target in enumerate(TARGET_COLS):
        model = _build_model(target, scale_pos_weights[target], seed)
        model.fit(x_train_flat[target], y_train[:, i])
        final_models[target] = model
        proba = model.predict_proba(x_test_flat[target])[:, 1]
        proba_test[target] = proba
        pred_test[target] = (proba >= final_thresholds[target]).astype(int)

    # -- Bootstrap bagging for uncertainty (Fase 5b) ----------------------------
    rng = np.random.RandomState(seed)
    bagging_models: dict[str, list] = {t: [] for t in TARGET_COLS}
    for i, target in enumerate(TARGET_COLS):
        x_tr_t, y_tr_t = x_train_flat[target], y_train[:, i]
        n_tr = len(y_tr_t)
        for b in range(n_bagging):
            boot_idx = rng.randint(0, n_tr, size=n_tr)
            x_boot, y_boot = x_tr_t[boot_idx], y_tr_t[boot_idx]
            if len(np.unique(y_boot)) < 2:
                continue  # muestra bootstrap degenerada (una sola clase) -- se omite
            seed_b = seed + b + 1
            if target == "target_animales":
                n_neg_b = int((y_boot == 0).sum())
                n_pos_b = int((y_boot == 1).sum())
                spw_b = n_neg_b / n_pos_b if n_pos_b > 0 else 1.0
                bmodel = _build_xgb(spw_b, seed_b)
            else:
                bmodel = _build_logreg(seed_b)
            bmodel.fit(x_boot, y_boot)
            bagging_models[target].append(bmodel)

    # -- Test evaluation ----------------------------------------------------------
    metrics: dict[str, dict[str, float]] = {}
    for i, target in enumerate(TARGET_COLS):
        metrics[target] = _evaluate(y_test[:, i], pred_test[target], proba_test[target])
        metrics[target]["threshold"] = final_thresholds[target]

    logger.info(
        "ml16 train_models() done — n_train=%d n_test=%d animales_f1=%.3f insumos_f1=%.3f",
        len(y_train), len(y_test), metrics["target_animales"]["f1"], metrics["target_insumos"]["f1"],
    )
    return {
        "models": final_models,
        "scalers": scalers,
        "bagging_models": bagging_models,
        "thresholds": final_thresholds,
        "metrics": metrics,
        "input_cols_per_target": input_cols_per_target,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }
