"""Canonical metric calculations for the official CU28 pipeline."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


def percentage_reduction(
    baseline: float | pd.Series | np.ndarray,
    candidate: float | pd.Series | np.ndarray,
) -> float | np.ndarray:
    """Return `(baseline - candidate) / baseline * 100` with canonical zero handling."""

    baseline_array = np.asarray(baseline, dtype=float)
    candidate_array = np.asarray(candidate, dtype=float)
    result = np.full(np.broadcast_shapes(baseline_array.shape, candidate_array.shape), np.nan, dtype=float)
    baseline_broadcast, candidate_broadcast = np.broadcast_arrays(baseline_array, candidate_array)
    positive_baseline = baseline_broadcast > 0
    np.divide(
        baseline_broadcast - candidate_broadcast,
        baseline_broadcast,
        out=result,
        where=positive_baseline,
    )
    result *= 100.0
    zero_baseline = ~positive_baseline
    result[zero_baseline & (candidate_broadcast <= 0)] = 0.0
    result[zero_baseline & (candidate_broadcast > 0)] = -100.0
    return float(result) if result.ndim == 0 else result


def percentage_change(
    baseline: float | pd.Series | np.ndarray,
    candidate: float | pd.Series | np.ndarray,
) -> float | np.ndarray:
    """Return `(candidate - baseline) / baseline * 100` with canonical zero handling."""

    baseline_array = np.asarray(baseline, dtype=float)
    candidate_array = np.asarray(candidate, dtype=float)
    result = np.full(np.broadcast_shapes(baseline_array.shape, candidate_array.shape), np.nan, dtype=float)
    baseline_broadcast, candidate_broadcast = np.broadcast_arrays(baseline_array, candidate_array)
    positive_baseline = baseline_broadcast > 0
    np.divide(
        candidate_broadcast - baseline_broadcast,
        baseline_broadcast,
        out=result,
        where=positive_baseline,
    )
    result *= 100.0
    zero_baseline = ~positive_baseline
    result[zero_baseline & (candidate_broadcast <= 0)] = 0.0
    result[zero_baseline & (candidate_broadcast > 0)] = 100.0
    return float(result) if result.ndim == 0 else result


def stockout_guardrail_pass(
    baseline_stockout: float | pd.Series | np.ndarray,
    policy_stockout: float | pd.Series | np.ndarray,
    *,
    allowed_increase_pct: float,
) -> bool | np.ndarray:
    """Apply the same bounded-stockout guardrail in every simulation path."""

    baseline_array, policy_array = np.broadcast_arrays(
        np.asarray(baseline_stockout, dtype=float),
        np.asarray(policy_stockout, dtype=float),
    )
    result = np.where(
        baseline_array > 0,
        policy_array <= baseline_array * (1.0 + allowed_increase_pct / 100.0) + 1e-9,
        policy_array <= baseline_array + 1e-9,
    )
    return bool(result) if result.ndim == 0 else result


def _aligned_numeric_pairs(y_true: Iterable[Any], y_pred: Iterable[Any]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "actual": pd.to_numeric(pd.Series(y_true), errors="coerce"),
            "prediction": pd.to_numeric(pd.Series(y_pred), errors="coerce"),
        }
    ).dropna()


def compute_regression_metrics(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    *,
    split: str,
    model: str,
    target: str,
    include_mape: bool = True,
    include_distribution: bool = True,
) -> dict[str, Any]:
    """Compute the canonical regression contract from aligned finite pairs."""

    aligned = _aligned_numeric_pairs(y_true, y_pred)
    if aligned.empty:
        return {
            "split": split,
            "model": model,
            "target": target,
            "n_samples": 0,
            "rows": 0,
        }

    actual = aligned["actual"].astype(float)
    prediction = aligned["prediction"].astype(float)
    residual = prediction - actual
    squared_error = np.square(residual)
    total_variance = float(np.sum(np.square(actual - float(actual.mean()))))

    payload: dict[str, Any] = {
        "split": split,
        "model": model,
        "target": target,
        "n_samples": int(len(aligned)),
        "rows": int(len(aligned)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(squared_error))),
        "r2": float(1.0 - float(np.sum(squared_error)) / total_variance) if total_variance > 0 else 0.0,
    }

    if include_mape:
        non_zero = actual != 0
        payload["mape"] = (
            float(np.mean(np.abs((actual[non_zero] - prediction[non_zero]) / actual[non_zero])))
            if non_zero.any()
            else float("nan")
        )

    if include_distribution:
        actual_std = float(actual.std(ddof=0))
        prediction_std = float(prediction.std(ddof=0))
        actual_variance = float(actual.var(ddof=0))
        payload.update(
            {
                "prediction_bias": float(residual.mean()),
                "prediction_mean": float(prediction.mean()),
                "actual_mean": float(actual.mean()),
                "prediction_std": prediction_std,
                "actual_std": actual_std,
                "variance_ratio": float(prediction_std / actual_std) if actual_std > 0 else float("nan"),
                "slope_pred_vs_actual": (
                    float(np.cov(actual, prediction, ddof=0)[0, 1] / actual_variance)
                    if actual_variance > 0
                    else float("nan")
                ),
                "underprediction_rate": float((prediction < actual).mean()),
            }
        )
    return payload


def compute_trigger_metrics(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    y_proba: Iterable[Any] | None = None,
    *,
    split: str,
) -> dict[str, Any]:
    """Compute the canonical binary-classification contract.

    BUY is class 1 and DO_NOT_BUY is class 0.
    """

    aligned = pd.DataFrame(
        {
            "actual": pd.to_numeric(pd.Series(y_true), errors="coerce"),
            "prediction": pd.to_numeric(pd.Series(y_pred), errors="coerce"),
        }
    ).dropna()
    if aligned.empty:
        return {"split": split, "n_samples": 0, "rows": 0}

    actual = aligned["actual"].astype(int)
    prediction = aligned["prediction"].astype(int)
    true_positive = int(((actual == 1) & (prediction == 1)).sum())
    true_negative = int(((actual == 0) & (prediction == 0)).sum())
    false_positive = int(((actual == 0) & (prediction == 1)).sum())
    false_negative = int(((actual == 1) & (prediction == 0)).sum())
    support_buy = int((actual == 1).sum())
    support_do_not_buy = int((actual == 0).sum())
    total = int(len(aligned))

    precision_buy = true_positive / max(true_positive + false_positive, 1)
    precision_do_not_buy = true_negative / max(true_negative + false_negative, 1)
    recall_buy = true_positive / max(support_buy, 1)
    recall_do_not_buy = true_negative / max(support_do_not_buy, 1)
    f1_buy = (
        0.0
        if precision_buy + recall_buy == 0
        else 2.0 * precision_buy * recall_buy / (precision_buy + recall_buy)
    )
    f1_do_not_buy = (
        0.0
        if precision_do_not_buy + recall_do_not_buy == 0
        else 2.0
        * precision_do_not_buy
        * recall_do_not_buy
        / (precision_do_not_buy + recall_do_not_buy)
    )

    payload: dict[str, Any] = {
        "split": split,
        "n_samples": total,
        "rows": total,
        "positive_rate_actual": float((actual == 1).mean()),
        "positive_rate_pred": float((prediction == 1).mean()),
        "accuracy": float((true_positive + true_negative) / max(total, 1)),
        "balanced_accuracy": float((recall_buy + recall_do_not_buy) / 2.0),
        "precision_by_class": {
            "BUY": float(precision_buy),
            "DO_NOT_BUY": float(precision_do_not_buy),
        },
        "recall_buy": float(recall_buy),
        "recall_do_not_buy": float(recall_do_not_buy),
        "f1_buy": float(f1_buy),
        "f1_do_not_buy": float(f1_do_not_buy),
        "false_negative_rate": float(false_negative / max(support_buy, 1)),
        "support_by_class": {
            "BUY": support_buy,
            "DO_NOT_BUY": support_do_not_buy,
        },
        "confusion_matrix": {
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_positive": true_positive,
        },
        # Backward-compatible BUY aliases used by existing reports.
        "precision": float(precision_buy),
        "recall": float(recall_buy),
        "f1": float(f1_buy),
    }
    if y_proba is not None:
        probabilities = pd.to_numeric(pd.Series(y_proba), errors="coerce").dropna()
        payload["probability_summary"] = {
            "mean": float(probabilities.mean()),
            "std": float(probabilities.std(ddof=0) if len(probabilities) > 1 else 0.0),
            "min": float(probabilities.min()),
            "max": float(probabilities.max()),
        }
    return payload


def compute_policy_metrics(
    period_df: pd.DataFrame,
    *,
    guardrail_name: str,
    allowed_stockout_increase_pct: float,
) -> dict[str, Any]:
    """Aggregate the canonical policy metrics from period-level results."""

    baseline_excess = float(pd.to_numeric(period_df["baseline_excess_tons"], errors="coerce").sum())
    policy_excess = float(pd.to_numeric(period_df["excess_tons"], errors="coerce").sum())
    baseline_stockout = float(pd.to_numeric(period_df["baseline_stockout_tons"], errors="coerce").sum())
    policy_stockout = float(pd.to_numeric(period_df["stockout_tons"], errors="coerce").sum())
    absolute_reduction = baseline_excess - policy_excess
    return {
        "n_periods": int(len(period_df)),
        "rows": int(len(period_df)),
        "baseline_excess_tons": baseline_excess,
        "policy_excess_tons": policy_excess,
        "absolute_excess_reduction_tons": float(absolute_reduction),
        "aggregate_excess_reduction_pct": float(percentage_reduction(baseline_excess, policy_excess)),
        "baseline_stockout_tons": baseline_stockout,
        "policy_stockout_tons": policy_stockout,
        "stockout_tons": policy_stockout,
        "aggregate_stockout_change_pct": float(percentage_change(baseline_stockout, policy_stockout)),
        "guardrail": {
            "name": guardrail_name,
            "allowed_stockout_increase_pct": float(allowed_stockout_increase_pct),
        },
        "stockout_guardrail_pass": bool(
            stockout_guardrail_pass(
                baseline_stockout,
                policy_stockout,
                allowed_increase_pct=allowed_stockout_increase_pct,
            )
        ),
    }
