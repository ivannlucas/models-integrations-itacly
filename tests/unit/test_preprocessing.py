"""Unit tests for wine-sulphite preprocessing helpers."""
import types

import numpy as np
import pytest

from app.plugins.ml25_wine_sulphites.preprocessing import (
    FEATURES_BOUND,
    FEATURES_QUAL,
    build_simulation_grid,
    map_request_to_wine_dict,
)

_DEFAULTS = {
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.66,
    "citric_acid": 0.0,
    "residual_sugar": 1.8,
    "chlorides": 0.075,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
    "free_sulfur_dioxide": 11.0,
    "total_sulfur_dioxide": 34.0,
}


def _make_request(**overrides):
    """Return a SimpleNamespace with default wine properties, applying overrides."""
    return types.SimpleNamespace(**{**_DEFAULTS, **overrides})


def _base_wine():
    """Return a canonical base_wine dict with space-separated keys."""
    return map_request_to_wine_dict(_make_request())


# ── map_request_to_wine_dict ──────────────────────────────────────────────────

def test_map_returns_space_separated_keys():
    """All returned keys use space-separated names matching training column names."""
    wine = _base_wine()
    expected_keys = [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "density", "pH", "sulphates", "alcohol",
        "free sulfur dioxide", "total sulfur dioxide",
    ]
    for key in expected_keys:
        assert key in wine, f"Expected key '{key}' not found"


def test_map_values_match_input_attributes():
    """Values in the dict match the corresponding snake_case attributes of the request."""
    req = _make_request(pH=3.2, alcohol=11.5, free_sulfur_dioxide=20.0)
    wine = map_request_to_wine_dict(req)
    assert wine["pH"] == pytest.approx(3.2)
    assert wine["alcohol"] == pytest.approx(11.5)
    assert wine["free sulfur dioxide"] == pytest.approx(20.0)


def test_map_returns_exactly_eleven_keys():
    """Returned dict contains exactly 11 physicochemical feature keys."""
    assert len(_base_wine()) == 11


def test_map_total_sulfur_dioxide_maps_correctly():
    """total_sulfur_dioxide attribute maps to 'total sulfur dioxide' key."""
    req = _make_request(total_sulfur_dioxide=120.0)
    wine = map_request_to_wine_dict(req)
    assert wine["total sulfur dioxide"] == pytest.approx(120.0)


# ── build_simulation_grid ─────────────────────────────────────────────────────

def test_grid_first_point_equals_current_free():
    """First free SO2 target equals the wine's current free sulfur dioxide value."""
    base_wine = _base_wine()
    free_targets, _, _ = build_simulation_grid(base_wine, delta_max=10.0)
    assert free_targets[0] == pytest.approx(base_wine["free sulfur dioxide"])


def test_grid_last_point_does_not_exceed_current_plus_delta():
    """Last free SO2 target does not exceed current_free + delta_max."""
    base_wine = _base_wine()
    delta = 20.0
    free_targets, _, _ = build_simulation_grid(base_wine, delta_max=delta)
    assert free_targets[-1] <= base_wine["free sulfur dioxide"] + delta + 1e-6


def test_grid_step_is_one_mg_l():
    """Consecutive free SO2 targets are spaced 1 mg/L apart."""
    base_wine = _base_wine()
    free_targets, _, _ = build_simulation_grid(base_wine, delta_max=5.0)
    diffs = np.diff(free_targets)
    assert np.allclose(diffs, 1.0)


def test_qual_rows_columns_match_features_qual():
    """qual_rows DataFrame has exactly the FEATURES_QUAL columns in the correct order."""
    _, qual_rows, _ = build_simulation_grid(_base_wine(), delta_max=5.0)
    assert list(qual_rows.columns) == list(FEATURES_QUAL)


def test_bound_rows_columns_match_features_bound():
    """bound_rows DataFrame has exactly the FEATURES_BOUND columns in the correct order."""
    _, _, bound_rows = build_simulation_grid(_base_wine(), delta_max=5.0)
    assert list(bound_rows.columns) == list(FEATURES_BOUND)


def test_grid_arrays_have_consistent_length():
    """free_targets, qual_rows, and bound_rows all share the same length."""
    free_targets, qual_rows, bound_rows = build_simulation_grid(_base_wine(), delta_max=10.0)
    assert len(free_targets) == len(qual_rows) == len(bound_rows)


def test_grid_delta_zero_produces_single_point():
    """delta_max=0 produces a one-element grid with only the current free SO2 value."""
    free_targets, qual_rows, bound_rows = build_simulation_grid(_base_wine(), delta_max=0.0)
    assert len(free_targets) == 1
    assert len(qual_rows) == 1
    assert len(bound_rows) == 1


def test_grid_free_targets_appear_in_qual_rows():
    """The free sulfur dioxide column in qual_rows matches free_targets exactly."""
    base_wine = _base_wine()
    free_targets, qual_rows, _ = build_simulation_grid(base_wine, delta_max=5.0)
    assert np.allclose(qual_rows["free sulfur dioxide"].values, free_targets)


def test_grid_free_targets_appear_in_bound_rows():
    """The free sulfur dioxide column in bound_rows matches free_targets exactly."""
    base_wine = _base_wine()
    free_targets, _, bound_rows = build_simulation_grid(base_wine, delta_max=5.0)
    assert np.allclose(bound_rows["free sulfur dioxide"].values, free_targets)


def test_grid_physico_columns_are_constant_across_rows():
    """Physical property columns (not free SO2) are identical in every row of bound_rows."""
    base_wine = _base_wine()
    _, _, bound_rows = build_simulation_grid(base_wine, delta_max=5.0)
    for col in ["fixed acidity", "pH", "alcohol"]:
        assert bound_rows[col].nunique() == 1, f"Column '{col}' should be constant"
