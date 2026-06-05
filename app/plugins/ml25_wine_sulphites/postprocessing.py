"""Postprocessing functions for the wine sulphite plugin."""
import numpy as np

from app.plugins.ml25_wine_sulphites.preprocessing import PKA_SO2


def decode_bound_predictions(raw_bound: np.ndarray, free_targets: np.ndarray) -> np.ndarray:
    """Undo log1p transform and enforce physical monotonicity of bound SO2.

    Bound SO2 must be non-decreasing as free SO2 increases (physico-chemical law).

    Args:
        raw_bound: Raw model output in log1p space.
        free_targets: Corresponding free SO2 values (used to determine sort order).

    Returns:
        Array of bound SO2 values (mg/L), monotonically non-decreasing.
    """
    pred_bounds = np.maximum(np.expm1(raw_bound), 0.0)

    order = np.argsort(free_targets)
    pred_sorted = pred_bounds[order]
    pred_sorted = np.maximum.accumulate(pred_sorted)  # enforce monotonicity
    return pred_sorted[np.argsort(order)]


def compute_molecular_so2(free_targets: np.ndarray, ph: float) -> np.ndarray:
    """Calculate molecular SO2 concentration from free SO2 and wine pH.

    Formula: [SO2 molecular] = free_SO2 / (1 + 10^(pH - pKa))

    Args:
        free_targets: Array of free SO2 values (mg/L).
        ph: Wine pH.

    Returns:
        Array of molecular SO2 values (mg/L).
    """
    return free_targets / (1.0 + 10.0 ** (ph - PKA_SO2))


def apply_operational_constraints(
    free_targets: np.ndarray,
    pred_bounds: np.ndarray,
    pred_totals: np.ndarray,
    molecular_so2: np.ndarray,
    pred_qualities: np.ndarray,
    min_molecular: float,
    max_total: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Filter simulation points to those satisfying operational constraints.

    Args:
        free_targets: Candidate free SO2 values (mg/L).
        pred_bounds: Predicted bound SO2 values (mg/L).
        pred_totals: Predicted total SO2 values (mg/L).
        molecular_so2: Molecular SO2 values (mg/L).
        pred_qualities: Predicted sensory quality scores.
        min_molecular: Minimum molecular SO2 for microbial protection (mg/L).
        max_total: Maximum legal total SO2 (mg/L).

    Returns:
        Tuple of filtered arrays: (free, bound, total, molecular, quality).

    Raises:
        ValueError: If no simulation point satisfies the constraints.
    """
    valid_mask = (molecular_so2 >= min_molecular) & (pred_totals <= max_total)
    if not valid_mask.any():
        raise ValueError(
            f"No simulation point satisfies constraints: "
            f"min_molecular={min_molecular} mg/L, max_total={max_total} mg/L. "
            "Try relaxing delta_max or adjusting constraints."
        )
    return (
        free_targets[valid_mask],
        pred_bounds[valid_mask],
        pred_totals[valid_mask],
        molecular_so2[valid_mask],
        pred_qualities[valid_mask],
    )


def select_recommendation(
    _valid_free: np.ndarray,
    _valid_bounds: np.ndarray,
    _valid_totals: np.ndarray,
    _valid_moleculars: np.ndarray,
    valid_qualities: np.ndarray,
    baseline_quality: float,
    mae_quality: float,
) -> tuple[int, str, bool]:
    """Select the recommended simulation point using the 1×MAE threshold rule.

    If the best reachable quality improvement exceeds 1×MAE, recommend that point.
    Otherwise, recommend the minimum safe intervention (lowest valid free SO2).

    Args:
        valid_free: Filtered free SO2 values satisfying constraints.
        valid_bounds: Filtered bound SO2 values.
        valid_totals: Filtered total SO2 values.
        valid_moleculars: Filtered molecular SO2 values.
        valid_qualities: Filtered predicted quality scores.
        baseline_quality: Predicted quality at the current (unmodified) free SO2.
        mae_quality: Cross-validated MAE of the quality model (used as threshold).

    Returns:
        tuple: (rec_idx, reason, intervention_recommended)
    """
    best_idx = int(np.argmax(valid_qualities))
    best_quality = float(valid_qualities[best_idx])
    gain = best_quality - baseline_quality
    threshold = mae_quality

    if gain > threshold:
        reason = (
            f"Significant improvement over baseline "
            f"(+{gain:.3f} > threshold {threshold:.3f}). "
            "Recommending maximum-benefit dose."
        )
        return best_idx, reason, True
    reason = (
        f"Improvement not significant (gain {gain:.3f} <= threshold {threshold:.3f}). "
        "Recommending minimum safe intervention."
    )
    return 0, reason, False
