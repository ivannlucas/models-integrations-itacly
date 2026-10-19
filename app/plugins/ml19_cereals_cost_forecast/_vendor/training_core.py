"""Verbatim copy of the official DATAGIA-19 training helpers (inbox/a19/codigo/src/).

Only the functions of the official pipeline (run_multihorizon_training) are copied, with their
bodies unchanged:

* src/utils/constants.py — targets, NON_FEATURES, TRAIN_END, MIN_CORR, horizons, hyperparameters
* src/training/train.py — build_multihorizon_targets, _select_best_model
* src/training/feature_selection.py — is_contemporary_return, audit_leakage, select_features
* src/training/models.py — eval_regression, eval_classification, best_lasso_alpha
* src/training/walk_forward.py — walk_forward_regression, walk_forward_classification

Left out on purpose: the DRNN (experimental, outside the official model ranking; it is what
makes models.py import torch), the legacy run_walk_forward, the reliability verdicts and the
report/CSV writers. src.utils.logging.get_logger is replaced by logging.getLogger.
"""
# pylint: disable=line-too-long,invalid-name  # verbatim vendored code
# flake8: noqa
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import (
    accuracy_score, brier_score_loss, f1_score, mean_absolute_error, mean_squared_error,
    r2_score, roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

# ── src/utils/constants.py ────────────────────────────────────────────────────

TARGET_COLS_RAW  = ["cebada", "trigo", "maiz"]
TARGET_COLS      = ["target_cebada", "target_trigo", "target_maiz"]
TARGET_REG       = "target_ret_next"
TARGET_CLF       = "target_dir_next"
NON_FEATURES     = [
    "date",
    "target_cebada", "target_trigo", "target_maiz",
    "target_media", "target_ret_next", "target_next", "target_dir_next",
    # Indices MAPA contemporaneos (sin lag): su valor "en T" no esta
    # genuinamente disponible en produccion (desfase de publicacion MAPA,
    # ver MAPA_PUBLICATION_LAG_MONTHS). Solo sus versiones _lagN son
    # seleccionables como features (idx_fertilizantes_lag3, etc.).
    "idx_fertilizantes", "idx_piensos", "idx_semillas", "idx_bienes_servicios",
    "idx_energia", "idx_bienes_inversion", "idx_fitosanitarios", "idx_gastos_generales",
    # Multi-horizonte (se anaden dinamicamente segun horizonte en select_features_strict)
    "ret_1m", "ret_2m", "ret_3m", "ret_4m", "ret_5m", "ret_6m", "ret_12m",
    "dir_1m", "dir_2m", "dir_3m", "dir_4m", "dir_5m", "dir_6m", "dir_12m",
]

TRAIN_END  = "2019-12-31"   # ultimo mes del train set
MIN_CORR   = 0.08           # umbral minimo de |r| en feature selection

REG_HORIZONS = [1, 2, 3]   # horizontes regresion: H1, H2, H3
CLF_HORIZONS = [1, 2, 3]   # horizontes clasificacion: H1, H2, H3

RIDGE_ALPHA = 100

XGB_PARAMS = dict(
    max_depth=2,
    learning_rate=0.1,
    subsample=0.7,
    colsample_bytree=0.6,
    reg_alpha=1.0,
    reg_lambda=5.0,
    n_estimators=200,
    random_state=42,
    verbosity=0,
)

LR_C       = 0.1           # LogisticRegression C
RF_PARAMS  = dict(n_estimators=200, max_depth=3, min_samples_leaf=10, random_state=42)
XGBC_PARAMS = dict(
    max_depth=2,
    learning_rate=0.1,
    subsample=0.7,
    colsample_bytree=0.6,
    reg_alpha=1.0,
    reg_lambda=5.0,
    n_estimators=200,
    eval_metric="logloss",
    random_state=42,
    verbosity=0,
)


# ── src/training/train.py ─────────────────────────────────────────────────────

def build_multihorizon_targets(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """
    Anade columnas ret_{h}m y dir_{h}m para cada horizonte h.
    Basado en notebook 03, seccion 1.
    """
    df = df.copy()
    for h in horizons:
        col_ret = f"ret_{h}m"
        col_dir = f"dir_{h}m"
        df[col_ret] = df["target_media"].pct_change(h).shift(-h) * 100
        df[col_dir] = (df[col_ret] > 0).astype(int)
    return df


def _select_best_model(available: dict[str, dict], metric: str, task: str) -> str:
    """Selecciona mejor modelo segun metrica; minimiza solo errores."""
    minimize_metrics = {"mae", "rmse", "brier"}
    if metric in minimize_metrics:
        return min(available, key=lambda m: available[m][metric])
    return max(available, key=lambda m: available[m][metric])


# ── src/training/feature_selection.py ─────────────────────────────────────────

_CONTEMPORARY_RET_RE = re.compile(r"^ret_(?!.*_lag)", re.IGNORECASE)


def is_contemporary_return(col: str) -> bool:
    """True si la columna es un retorno contemporaneo (excluir en STRICT)."""
    return bool(_CONTEMPORARY_RET_RE.match(col))


def audit_leakage(
    df: pd.DataFrame,
    all_features: list[str],
    threshold: float = 0.95,
) -> list[str]:
    """
    Detecta features con correlacion > threshold con 'target_next'
    (posible data leakage lineal). Las elimina y devuelve las features seguras.

    Importante:
      - Es una auditoria univariante y lineal (Pearson).
      - Umbral fijo por diseno (default=0.95) para estabilidad y reproducibilidad.
    """
    leakage = []
    for feat in all_features:
        if "target_next" not in df.columns:
            break
        r = df[feat].corr(df["target_next"])
        if not np.isnan(r) and abs(r) > threshold:
            log.warning(f"  [!]  LEAKAGE: {feat}  r_target_next = {r:.4f}")
            leakage.append(feat)

    safe = [f for f in all_features if f not in leakage and "target_next" not in f.lower()]
    log.info(f"Auditoria leakage: {len(leakage)} eliminadas -> {len(safe)} features seguras")
    return safe


def select_features(
    df: pd.DataFrame,
    min_corr: float = MIN_CORR,
    max_inter_corr: float = 0.90,
    target_col: str = TARGET_REG,
    strict: bool = False,
    leakage_threshold: float = 0.95,
    train_mask: pd.Series | None = None,
) -> tuple[list[str], dict[str, float]]:
    """
    Seleccion completa adaptada al notebook 03 (multi-horizonte).

    Parametros
    ----------
    df          : DataFrame completo (o solo train si train_mask=None y no hay test)
    min_corr    : umbral minimo de |r| con el target
    max_inter_corr : umbral de correlacion inter-feature para eliminar redundancias
    target_col  : columna objetivo
    strict      : si True -> excluye retornos contemporaneos (escenario produccion STRICT)
    leakage_threshold : umbral de auditoria de leakage lineal contra target_next
    train_mask  : mascara booleana sobre df para calcular correlaciones solo en train.
                  Si es None se usa TODO df (permite llamar con solo train).

    Devuelve
    --------
    (final_features, corr_dict)
    """
    # -- Paso 0: features candidatas ------------------------------------------
    all_features = [
        c for c in df.columns
        if c not in NON_FEATURES
        and c != target_col
        and not c.startswith("dir_")   # excluir targets de clasificacion
        and not c.startswith("ret_1m") # excluir targets de regresion multi-h
        and not c.startswith("ret_2m")
        and not c.startswith("ret_3m")
        and not c.startswith("ret_6m")
        and not c.startswith("ret_12m")
    ]

    # STRICT: excluir retornos contemporaneos
    if strict:
        contemporary = [f for f in all_features if is_contemporary_return(f)]
        if contemporary:
            log.info(f"STRICT: excluyendo {len(contemporary)} retornos contemporaneos: "
                     f"{contemporary[:5]}{'...' if len(contemporary) > 5 else ''}")
        all_features = [f for f in all_features if f not in contemporary]

    # sub-df sobre el que calcular estadisticos (solo train)
    df_sel = df.loc[train_mask] if train_mask is not None else df
    df_sel = df_sel.dropna(subset=[target_col])

    # Auditoria de leakage sobre train (heuristica lineal / umbral fijo configurable)
    if "target_next" in df_sel.columns:
        log.info(
            "Leakage audit (lineal): threshold=%.2f sobre proxy 'target_next' (solo train).",
            leakage_threshold,
        )
    else:
        log.info(
            "Leakage audit omitida: no existe columna proxy 'target_next' en el dataset de seleccion."
        )
    safe_features = audit_leakage(df_sel, all_features, threshold=leakage_threshold)

    # -- Paso 1: correlacion con target ---------------------------------------
    corr_dict: dict[str, float] = {}
    for feat in safe_features:
        if feat not in df_sel.columns:
            continue
        r = df_sel[feat].corr(df_sel[target_col])
        corr_dict[feat] = float(r) if not np.isnan(r) else 0.0

    corr_series = pd.Series(corr_dict).sort_values(key=abs, ascending=False)
    selected = corr_series[abs(corr_series) >= min_corr]
    log.info(
        f"Features |r| >= {min_corr}: {len(selected)} de {len(safe_features)}"
        f"{'  [STRICT]' if strict else ''}"
    )

    # -- Paso 2: eliminar redundancia -----------------------------------------
    selected_feats = list(selected.index)
    # Calcular la matriz de correlacion solo en train
    corr_matrix = df_sel[selected_feats].corr()

    to_drop: set[str] = set()
    for i in range(len(selected_feats)):
        if selected_feats[i] in to_drop:
            continue
        for j in range(i + 1, len(selected_feats)):
            if selected_feats[j] in to_drop:
                continue
            val = corr_matrix.iloc[i, j]
            if not np.isnan(val) and abs(val) > max_inter_corr:
                fi, fj = selected_feats[i], selected_feats[j]
                drop = fj if abs(corr_dict.get(fi, 0)) >= abs(corr_dict.get(fj, 0)) else fi
                to_drop.add(drop)

    final_features = [f for f in selected_feats if f not in to_drop]
    log.info(f"Tras eliminar redundancia: {len(final_features)} features finales")

    return final_features, corr_dict


# ── src/training/models.py ────────────────────────────────────────────────────

def eval_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str = "",
) -> dict:
    """
    Evalua un modelo de regresion para series temporales con senal de trading.

    Metricas: MAE, RMSE, R², correlacion de Pearson, accuracy direccional,
    Sharpe (senal long/short, anualizado x√12) y DirAcc en alta confianza.
    """
    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2    = r2_score(y_true, y_pred)
    corr  = float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_pred) > 0 else 0.0

    dir_pred = (y_pred > 0).astype(int)
    dir_true = (y_true > 0).astype(int)
    dir_acc  = accuracy_score(dir_true, dir_pred)

    sig_ret  = np.sign(y_pred) * y_true
    sharpe   = (sig_ret.mean() / sig_ret.std() * np.sqrt(12)
                if sig_ret.std() > 0 else 0.0)

    med_pred = float(np.median(np.abs(y_pred)))
    if med_pred > 0:
        hc = np.abs(y_pred) > med_pred
        dir_acc_hc = float(accuracy_score(dir_true[hc], dir_pred[hc])) if hc.sum() > 0 else float("nan")
    else:
        dir_acc_hc = float("nan")

    return dict(
        name=name, mae=mae, rmse=rmse, r2=r2, corr=corr,
        dir_acc=dir_acc, sharpe=sharpe, dir_acc_high_conf=dir_acc_hc,
    )


def eval_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    name: str = "",
) -> dict:
    """
    Evalua un modelo de clasificacion binaria (direccion de precio).

    Metricas: accuracy, F1, ROC-AUC, Brier score, Sharpe de senal L/S.

    Parametros
    ----------
    y_true : etiquetas reales (0/1)
    y_pred : etiquetas predichas (0/1)
    probs  : probabilidad de clase 1 (P(sube))
    name   : nombre del modelo para el dict result
    """
    acc = float(accuracy_score(y_true, y_pred))
    f1  = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, probs))
    except Exception:
        auc = 0.5
    bri = float(brier_score_loss(y_true, probs))

    # Sharpe: senal larga si pred=1, corta si pred=0; acierto +1, error -1
    sig = np.where(y_pred == 1, 1.0, -1.0) * (2.0 * y_true.astype(float) - 1.0)
    sharpe = float(sig.mean() / sig.std() * np.sqrt(12)) if sig.std() > 0 else 0.0

    return dict(
        name=name, acc=acc, f1=f1, auc=auc, brier=bri, sharpe=sharpe,
        # guardamos arrays para analisis posterior (distribucion, equity, etc.)
        preds=y_pred, y_test=y_true, probs=probs,
    )


def best_lasso_alpha(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alphas: list[float] | None = None,
    n_splits: int = 5,
) -> float:
    """
    Selecciona el alpha optimo de Lasso maximizando correlacion de Pearson
    en TimeSeriesSplit. Devuelve best_alpha.
    """
    if alphas is None:
        # Merged range: covers nb03 [0.05..2.0] plus broader exploration
        alphas = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_alpha, best_score = alphas[0], -np.inf

    for alpha in alphas:
        scores = []
        for tr_idx, val_idx in tscv.split(X_train):
            mdl = Lasso(alpha=alpha, max_iter=20000)
            mdl.fit(X_train[tr_idx], y_train[tr_idx])
            pred = mdl.predict(X_train[val_idx])
            if np.std(pred) == 0:
                scores.append(0.0)
                continue
            r = float(np.corrcoef(y_train[val_idx], pred)[0, 1])
            scores.append(r if not np.isnan(r) else 0.0)
        mean_r = float(np.mean(scores))
        if mean_r > best_score:
            best_score, best_alpha = mean_r, alpha

    log.info(f"Lasso: best_alpha={best_alpha}  CV_r={best_score:+.3f}")
    return best_alpha


# ── src/training/walk_forward.py ──────────────────────────────────────────────

def walk_forward_regression(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    train_end: str,
    model_factory: Callable,
    verbose: bool = False,
    trace_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Walk-Forward Regression con expanding window.

    En cada paso:
      - Ajusta StandardScaler solo sobre los datos de train acumulados
      - Entrena un modelo fresh con model_factory()
      - Predice el siguiente mes

    Parametros
    ----------
    df            : DataFrame con features + target_col + 'date'
    features      : lista de features (sin fugas)
    target_col    : nombre de la columna target (ret_{h}m)
    train_end     : fecha de corte train/test ('2019-12-31')
    model_factory : callable sin args -> instancia de modelo sklearn

    Devuelve
    --------
    (y_test, preds, dates_test)

    Si `trace_path` se proporciona, guarda un CSV con el detalle de cada
    iteracion walk-forward (tamano de train, fecha de test, y_true, y_pred).
    """
    df_c = df.dropna(subset=features + [target_col]).copy()
    X    = df_c[features].values
    y    = df_c[target_col].values
    dates = df_c["date"].values

    idx0   = int((df_c["date"] <= train_end).sum())
    n_test = len(y) - idx0

    if n_test <= 0:
        log.warning(f"WF Reg [{target_col}]: sin datos de test tras {train_end}")
        return np.array([]), np.array([]), np.array([])

    preds = np.zeros(n_test)
    trace_rows: list[dict] = []

    for i in range(n_test):
        t = idx0 + i
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X[:t])
        X_te_s = sc.transform(X[t: t + 1])

        mdl = model_factory()
        mdl.fit(X_tr_s, y[:t])
        preds[i] = float(mdl.predict(X_te_s)[0])
        trace_rows.append({
            "wf_step": int(i),
            "train_end_index_exclusive": int(t),
            "train_size": int(t),
            "test_date": pd.to_datetime(dates[t]),
            "y_true": float(y[t]),
            "y_pred": float(preds[i]),
        })

        if verbose and (i + 1) % 20 == 0:
            log.info(f"  WF Reg [{target_col}] paso {i + 1}/{n_test}")

    if trace_path is not None:
        trace_df = pd.DataFrame(trace_rows)
        trace_out = Path(trace_path)
        trace_out.parent.mkdir(parents=True, exist_ok=True)
        trace_df.to_csv(trace_out, index=False)

    return y[idx0:], preds, dates[idx0:]


def walk_forward_classification(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    train_end: str,
    model_factory: Callable,
    verbose: bool = False,
    trace_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Walk-Forward Classification con expanding window.

    Parametros iguales a walk_forward_regression.

    Devuelve
    --------
    (y_test, preds, probs, dates_test)
    probs : probabilidad de clase 1 (P(sube)) de predict_proba

    Si `trace_path` se proporciona, guarda un CSV con el detalle de cada
    iteracion walk-forward (tamano de train, fecha de test, y_true, y_pred, prob_up).
    """
    df_c  = df.dropna(subset=features + [target_col]).copy()
    X     = df_c[features].values
    y     = df_c[target_col].values.astype(int)
    dates = df_c["date"].values

    idx0   = int((df_c["date"] <= train_end).sum())
    n_test = len(y) - idx0

    if n_test <= 0:
        log.warning(f"WF Clf [{target_col}]: sin datos de test tras {train_end}")
        return np.array([]), np.array([], dtype=int), np.array([]), np.array([])

    preds = np.zeros(n_test, dtype=int)
    probs = np.zeros(n_test)
    trace_rows: list[dict] = []

    for i in range(n_test):
        t = idx0 + i
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X[:t])
        X_te_s = sc.transform(X[t: t + 1])

        mdl = model_factory()
        mdl.fit(X_tr_s, y[:t])
        preds[i] = int(mdl.predict(X_te_s)[0])
        # predict_proba: columna 1 = P(clase=1)
        try:
            probs[i] = float(mdl.predict_proba(X_te_s)[0][1])
        except AttributeError:
            # modelos sin predict_proba (no deberia ocurrir con LR/RF/XGB)
            probs[i] = float(preds[i])
        trace_rows.append({
            "wf_step": int(i),
            "train_end_index_exclusive": int(t),
            "train_size": int(t),
            "test_date": pd.to_datetime(dates[t]),
            "y_true": int(y[t]),
            "y_pred": int(preds[i]),
            "prob_up": float(probs[i]),
        })

        if verbose and (i + 1) % 20 == 0:
            log.info(f"  WF Clf [{target_col}] paso {i + 1}/{n_test}")

    if trace_path is not None:
        trace_df = pd.DataFrame(trace_rows)
        trace_out = Path(trace_path)
        trace_out.parent.mkdir(parents=True, exist_ok=True)
        trace_df.to_csv(trace_out, index=False)

    return y[idx0:], preds, probs, dates[idx0:]
