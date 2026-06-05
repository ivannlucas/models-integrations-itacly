"""Postprocessing functions for the wine sulphite plugin."""
import numpy as np

from app.plugins.wine_sulphite.preprocessing import PKA_SO2


def decode_bound_predictions(raw_bound: np.ndarray, free_targets: np.ndarray) -> np.ndarray:
    """Undo log1p transform and enforce physical monotonicity of bound SO2."""
    pred_bounds = np.maximum(np.expm1(raw_bound), 0.0)

    order = np.argsort(free_targets)
    pred_sorted = pred_bounds[order]
    pred_sorted = np.maximum.accumulate(pred_sorted)  # enforce monotonicity
    return pred_sorted[np.argsort(order)]


def compute_molecular_so2(free_targets: np.ndarray, pH: float) -> np.ndarray:
    """Calculate molecular SO2 concentration from free SO2 and wine pH."""
    return free_targets / (1.0 + 10.0 ** (pH - PKA_SO2))


def apply_operational_constraints(
    free_targets: np.ndarray,
    pred_bounds: np.ndarray,
    pred_totals: np.ndarray,
    molecular_so2: np.ndarray,
    pred_qualities: np.ndarray,
    min_molecular: float,
    max_total: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Filter simulation points to those satisfying operational constraints."""
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
    valid_free: np.ndarray,
    valid_bounds: np.ndarray,
    valid_totals: np.ndarray,
    valid_moleculars: np.ndarray,
    valid_qualities: np.ndarray,
    baseline_quality: float,
    mae_quality: float,
) -> tuple[int, str, bool]:
    """Select the recommended simulation point using the 1×MAE threshold rule."""
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
    else:
        reason = (
            f"Improvement not significant (gain {gain:.3f} <= threshold {threshold:.3f}). "
            "Recommending minimum safe intervention."
        )
        return 0, reason, False
