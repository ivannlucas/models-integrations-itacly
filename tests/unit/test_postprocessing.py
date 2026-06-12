"""Unit tests for wine-sulphite postprocessing helpers."""
import numpy as np
import pytest

from app.plugins.ml25_wine_sulphites.postprocessing import (
    apply_operational_constraints,
    compute_molecular_so2,
    decode_bound_predictions,
    select_recommendation,
)
from app.plugins.ml25_wine_sulphites.preprocessing import PKA_SO2


# ── decode_bound_predictions ──────────────────────────────────────────────────

def test_decode_output_length_matches_input():
    """Output array length equals the length of the input arrays."""
    raw = np.array([0.1, 0.2, 0.3, 0.4])
    free = np.arange(10.0, 14.0)
    assert len(decode_bound_predictions(raw, free)) == 4


def test_decode_all_values_non_negative():
    """All decoded bound SO2 values are non-negative, including clipped negatives."""
    raw = np.array([-5.0, -1.0, 0.0, np.log1p(10.0)])
    free = np.arange(10.0, 14.0)
    result = decode_bound_predictions(raw, free)
    assert np.all(result >= 0.0)


def test_decode_output_is_monotonically_non_decreasing():
    """Decoded bound values are monotonically non-decreasing with increasing free SO2."""
    raw = np.array([np.log1p(20.0), np.log1p(5.0), np.log1p(30.0)])
    free = np.array([10.0, 20.0, 30.0])
    result = decode_bound_predictions(raw, free)
    assert np.all(np.diff(result) >= 0.0)


def test_decode_zero_raw_gives_zero_bound():
    """A raw prediction of 0 (log1p(0)) decodes to exactly 0 bound SO2."""
    raw = np.array([0.0, 0.0])
    free = np.array([10.0, 20.0])
    result = decode_bound_predictions(raw, free)
    assert np.allclose(result, 0.0)


def test_decode_known_log1p_value():
    """expm1(log1p(x)) == x for a known positive input."""
    value = 15.0
    raw = np.array([np.log1p(value), np.log1p(value + 5)])
    free = np.array([10.0, 20.0])
    result = decode_bound_predictions(raw, free)
    assert result[0] == pytest.approx(value, rel=1e-5)


# ── compute_molecular_so2 ─────────────────────────────────────────────────────

def test_molecular_so2_known_value():
    """Verify result against a manually computed reference value."""
    free = np.array([20.0])
    pH = 3.2
    expected = 20.0 / (1.0 + 10.0 ** (pH - PKA_SO2))
    assert compute_molecular_so2(free, pH)[0] == pytest.approx(expected)


def test_molecular_so2_is_less_than_free_at_typical_wine_ph():
    """Molecular SO2 is always strictly less than free SO2 at typical wine pH."""
    free = np.array([10.0, 20.0, 30.0])
    result = compute_molecular_so2(free, ph=3.5)
    assert np.all(result < free)


def test_lower_ph_gives_higher_molecular_fraction():
    """Lower pH produces more molecular SO2 from the same free SO2 concentration."""
    free = np.array([20.0])
    assert compute_molecular_so2(free, ph=3.0)[0] > compute_molecular_so2(free, ph=3.8)[0]


def test_molecular_so2_zero_free_gives_zero():
    """Zero free SO2 always yields zero molecular SO2."""
    free = np.array([0.0])
    assert compute_molecular_so2(free, ph=3.5)[0] == pytest.approx(0.0)


def test_molecular_so2_output_length_matches_input():
    """Output array has the same length as the free_targets input."""
    free = np.array([10.0, 20.0, 30.0, 40.0])
    assert len(compute_molecular_so2(free, ph=3.5)) == 4


# ── apply_operational_constraints ────────────────────────────────────────────

def _arrays(n: int = 5):
    """Build default valid test arrays of length n."""
    free = np.linspace(10.0, 50.0, n)
    bounds = np.linspace(20.0, 60.0, n)
    totals = free + bounds
    molecular = free * 0.03  # ~0.3 – 1.5 mg/L
    qualities = np.linspace(5.0, 7.0, n)
    return free, bounds, totals, molecular, qualities


def test_constraints_returns_five_arrays():
    """Return value is a tuple of exactly five filtered arrays."""
    result = apply_operational_constraints(*_arrays(), min_molecular=0.0, max_total=200.0)
    assert len(result) == 5


def test_constraints_all_returned_arrays_have_same_length():
    """All five returned arrays have the same length after filtering."""
    result = apply_operational_constraints(*_arrays(), min_molecular=0.0, max_total=200.0)
    lengths = {len(a) for a in result}
    assert len(lengths) == 1


def test_constraints_passes_all_points_when_no_filtering():
    """All n points are returned when no constraint would filter any of them."""
    free, bounds, totals, molecular, qualities = _arrays(5)
    result = apply_operational_constraints(free, bounds, totals, molecular, qualities,
                                           min_molecular=0.0, max_total=1000.0)
    assert len(result[0]) == 5


def test_constraints_filters_below_min_molecular():
    """Points with molecular SO2 below min_molecular are excluded from the result."""
    free, bounds, totals, molecular, qualities = _arrays(5)
    # Threshold just above the second-highest molecular value → only 1 point survives
    threshold = float(np.sort(molecular)[-2]) + 0.001
    result = apply_operational_constraints(free, bounds, totals, molecular, qualities,
                                           min_molecular=threshold, max_total=200.0)
    assert len(result[0]) == 1


def test_constraints_filters_above_max_total():
    """Points with total SO2 above max_total are excluded from the result."""
    free, bounds, totals, molecular, qualities = _arrays(5)
    # Ceiling just below the second-lowest total → only 1 point survives
    ceiling = float(np.sort(totals)[1]) - 0.001
    result = apply_operational_constraints(free, bounds, totals, molecular, qualities,
                                           min_molecular=0.0, max_total=ceiling)
    assert len(result[0]) == 1


def test_constraints_raises_when_no_valid_points():
    """ValueError with 'No simulation point' is raised when all points fail constraints."""
    free = np.array([10.0])
    bounds = np.array([20.0])
    totals = np.array([30.0])
    molecular = np.array([0.1])
    qualities = np.array([5.0])
    with pytest.raises(ValueError, match="No simulation point"):
        apply_operational_constraints(free, bounds, totals, molecular, qualities,
                                      min_molecular=1.0, max_total=10.0)


# ── select_recommendation ─────────────────────────────────────────────────────

def _select(qualities: np.ndarray, baseline: float, mae: float):
    """Convenience wrapper that passes dummy arrays for free/bound/total/molecular."""
    dummy = np.zeros(len(qualities))
    return select_recommendation(dummy, dummy, dummy, dummy, qualities, baseline, mae)


def test_select_recommends_best_index_when_gain_exceeds_mae():
    """The best quality index is returned when gain > MAE threshold."""
    qualities = np.array([5.0, 5.5, 6.5])
    rec_idx, _, intervention = _select(qualities, baseline=5.0, mae=0.5)
    assert intervention is True
    assert rec_idx == int(np.argmax(qualities))


def test_select_recommends_index_zero_when_gain_does_not_exceed_mae():
    """Index 0 (minimum safe dose) is returned when gain <= MAE threshold."""
    qualities = np.array([5.0, 5.1, 5.2])
    rec_idx, _, intervention = _select(qualities, baseline=5.0, mae=0.5)
    assert intervention is False
    assert rec_idx == 0


def test_select_no_intervention_when_gain_exactly_equals_mae():
    """Gain exactly equal to MAE does not trigger intervention (threshold uses strict >)."""
    qualities = np.array([5.0, 5.5])
    _, _, intervention = _select(qualities, baseline=5.0, mae=0.5)
    assert intervention is False


def test_select_reason_contains_word_threshold():
    """The reason string always references the word 'threshold'."""
    qualities = np.array([5.0, 6.0])
    _, reason, _ = _select(qualities, baseline=5.0, mae=0.5)
    assert "threshold" in reason.lower()


def test_select_intervention_true_returns_non_zero_index():
    """When intervention is recommended, the returned index is the argmax, not 0."""
    qualities = np.array([5.0, 5.1, 7.0])
    rec_idx, _, intervention = _select(qualities, baseline=5.0, mae=0.5)
    assert intervention is True
    assert rec_idx == 2


def test_select_negative_gain_gives_no_intervention():
    """When the best reachable quality is worse than baseline, no intervention is recommended."""
    qualities = np.array([4.5, 4.8])
    _, _, intervention = _select(qualities, baseline=5.0, mae=0.1)
    assert intervention is False
