"""Vendored (subset) from inbox/a28/codigo/src/evaluation/metrics.py.

Only percentage_reduction/percentage_change/stockout_guardrail_pass/compute_policy_metrics are
vendored — compute_regression_metrics/compute_trigger_metrics are used only by the mixed_context
ML training pipeline, not by platform_run (not served by this plugin).
"""
from __future__ import annotations

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


def compute_policy_metrics(
    period_df: pd.DataFrame,
    *,
    guardrail_name: str,
    allowed_stockout_increase_pct: float,
) -> dict:
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
