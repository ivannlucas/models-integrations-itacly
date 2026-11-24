"""Pure-function tests for ml37's real preprocessing/rules code (not FakePlugin) — resampling,
schema_v1 padding, scaling, cycle stats, and the neurosymbolic rule engine, including the
deliberately-replicated accumulator alert-collapse quirk (see rules.py module docstring and
inbox/a37/manifest.yaml known_issues)."""
import numpy as np
import pytest

from app.domain.services.exceptions import DataContractError
from app.plugins.ml37_wine_press_predictive_maintenance import preprocessing, rules
from app.plugins.ml37_wine_press_predictive_maintenance.constants import (
    F_SCHEMA,
    PHYSICAL_SENSORS,
)


def test_resample_interpolates_linearly():
    """A 2-point cycle resampled to 5 steps linearly interpolates each sensor."""
    t = np.array([0.0, 1.0])
    series = {name: np.array([0.0, 10.0]) for name in PHYSICAL_SENSORS}
    out = preprocessing.resample_cycle_to_fixed_length(t, series, timesteps=5)
    assert out.shape == (5, len(PHYSICAL_SENSORS))
    assert np.allclose(out[:, 0], [0.0, 2.5, 5.0, 7.5, 10.0])


def test_resample_single_sample_cycle_broadcasts_constant():
    """A single-reading cycle fills every output timestep with that one value."""
    t = np.array([3.0])
    series = {name: np.array([7.0]) for name in PHYSICAL_SENSORS}
    out = preprocessing.resample_cycle_to_fixed_length(t, series, timesteps=4)
    assert np.allclose(out, 7.0)


def test_resample_non_finite_sensor_raises_data_contract_error():
    """A NaN/Inf sensor value raises DataContractError naming the sensor (not a generic 500)."""
    t = np.array([0.0, 1.0])
    series = {name: np.array([0.0, 10.0]) for name in PHYSICAL_SENSORS}
    series["PS1"] = np.array([0.0, float("inf")])
    with pytest.raises(DataContractError, match="PS1"):
        preprocessing.resample_cycle_to_fixed_length(t, series, timesteps=5)


def test_resample_duplicate_timestamps_raises_data_contract_error():
    t = np.array([0.0, 0.0, 1.0])
    series = {name: np.array([0.0, 1.0, 10.0]) for name in PHYSICAL_SENSORS}
    with pytest.raises(DataContractError, match="duplicados"):
        preprocessing.resample_cycle_to_fixed_length(t, series, timesteps=5)


def test_build_schema_v1_tensor_pads_to_40_cols():
    """Physical sensors occupy slots 0-13; slots 14-39 (virtual + reserved) stay zero."""
    resampled = np.ones((10, len(PHYSICAL_SENSORS)), dtype=np.float32)
    out = preprocessing.build_schema_v1_tensor(resampled)
    assert out.shape == (10, F_SCHEMA)
    assert np.all(out[:, : len(PHYSICAL_SENSORS)] == 1.0)
    assert np.all(out[:, len(PHYSICAL_SENSORS):] == 0.0)


def test_apply_scaler_standardizes_and_guards_zero_scale():
    X = np.array([[[2.0, 5.0]]], dtype=np.float32)
    scaler = {"mean": [1.0, 5.0], "scale": [0.5, 0.0]}
    out = preprocessing.apply_scaler(X, scaler)
    assert np.isclose(out[0, 0, 0], 2.0)  # (2-1)/0.5
    assert np.isclose(out[0, 0, 1], 0.0)  # scale 0 guarded to 1.0 -> (5-5)/1


def test_compute_cycle_stats_mean_std_population():
    resampled = np.zeros((4, len(PHYSICAL_SENSORS)), dtype=np.float32)
    ts1_pos = PHYSICAL_SENSORS.index("TS1")
    resampled[:, ts1_pos] = [1.0, 2.0, 3.0, 4.0]
    stats = preprocessing.compute_cycle_stats(resampled)
    assert stats["TS1_mean"] == pytest.approx(2.5)
    assert stats["TS1_std"] == pytest.approx(np.std([1.0, 2.0, 3.0, 4.0]))  # ddof=0


def test_parse_readings_missing_sensor_raises_data_contract_error():
    readings = [{"t": 0.0, **{name: 1.0 for name in PHYSICAL_SENSORS if name != "PS1"}}]
    with pytest.raises(DataContractError, match="PS1"):
        preprocessing.parse_readings(readings)


def test_parse_readings_empty_raises_data_contract_error():
    with pytest.raises(DataContractError):
        preprocessing.parse_readings([])


# --- Neurosymbolic rule engine ------------------------------------------------------------


def test_cooler_rule_critical_above_crit_threshold():
    alerts = rules.check_rule_evidence({"TS1_mean": 40.0})
    assert any(a["component"] == "cooler" and a["severity"] == "CRITICAL" for a in alerts)


def test_cooler_rule_warning_between_thresholds():
    alerts = rules.check_rule_evidence({"TS1_mean": 38.0})
    assert any(a["component"] == "cooler" and a["severity"] == "WARNING" for a in alerts)


def test_cooler_rule_no_alert_below_warn_threshold():
    alerts = rules.check_rule_evidence({"TS1_mean": 30.0})
    assert not any(a["component"] == "cooler" for a in alerts)


def test_valve_rule_differential_pressure():
    alerts = rules.check_rule_evidence({"PS3_mean": 10.0, "PS4_mean": 9.5})  # dp=0.5 < crit 0.8
    assert any(a["component"] == "valve" and a["severity"] == "CRITICAL" for a in alerts)


def test_pump_rule_efficiency():
    alerts = rules.check_rule_evidence({"FS1_mean": 0.0025, "EPS1_mean": 1.0})  # eff<crit 0.0030
    assert any(a["component"] == "pump" and a["severity"] == "CRITICAL" for a in alerts)


def test_accumulator_mean_and_std_rules_both_fire_independently():
    """PS1_mean<120 (CRITICAL) and PS1_std>4.0 (WARNING) are two separate `if` blocks — both
    fire for the same component, unlike the other three rules."""
    alerts = rules.check_rule_evidence({"PS1_mean": 110.0, "PS1_std": 5.0})
    accumulator_alerts = [a for a in alerts if a["component"] == "accumulator"]
    assert len(accumulator_alerts) == 2
    assert {a["severity"] for a in accumulator_alerts} == {"CRITICAL", "WARNING"}


def test_accumulator_legacy_collapse_can_silently_drop_the_critical_alert():
    """The real, deliberately-replicated quirk: because PS1_std is checked AFTER PS1_mean, the
    dict-comprehension legacy view keeps the WARNING and silently drops the CRITICAL — but
    rule_evidence (not legacy) still carries both."""
    result = rules.evaluate_cycle_alerts(
        {"PS1_mean": 110.0, "PS1_std": 5.0}, predictions={},
    )
    accumulator_rule_evidence = [a for a in result["rule_evidence"] if a["component"] == "accumulator"]
    assert len(accumulator_rule_evidence) == 2

    accumulator_fused = [a for a in result["alerts"] if a["component"] == "accumulator"]
    assert len(accumulator_fused) == 1
    assert accumulator_fused[0]["severity"] == "WARNING"  # the CRITICAL mean-alert was lost


def test_check_model_no_downgrade_for_high_trust_component():
    """cooler trust=0.97 >= TRUST_THRESHOLD: no downgrade regardless of proba."""
    alerts = rules.check_model({"cooler": {"class": 3, "proba": 0.95}})
    assert alerts == [{
        "component": "cooler", "source": "model", "severity": "CRITICAL",
        "message": "Clase predicha=3 (proba=0.95, trust=97%)", "value": 0.95,
    }]


def test_check_model_suppresses_low_trust_low_confidence():
    """valve trust=0.27 < threshold and proba=0.30 < MODEL_SUPPRESS_PROBA: alert suppressed."""
    alerts = rules.check_model({"valve": {"class": 73, "proba": 0.30}})
    assert alerts == []


def test_check_model_downgrades_low_trust_high_confidence():
    """valve trust=0.27 < threshold but proba=0.60 >= suppress floor: severity downgraded one
    level (CRITICAL -> WARNING) instead of suppressed."""
    alerts = rules.check_model({"valve": {"class": 73, "proba": 0.60}})
    assert len(alerts) == 1
    assert alerts[0]["severity"] == "WARNING"


def test_fuse_alerts_agreement_escalates_one_level_above_max():
    """Rule WARNING + model CRITICAL for the same component -> EMERGENCY, not simple max()."""
    rule_alerts = [{"component": "pump", "severity": "WARNING", "message": "r"}]
    model_alerts = [{"component": "pump", "severity": "CRITICAL", "message": "m"}]
    fused = rules.fuse_alerts(rule_alerts, model_alerts)
    assert len(fused) == 1
    assert fused[0]["severity"] == "EMERGENCY"
    assert fused[0]["source"] == "combined"


def test_fuse_alerts_single_source_passes_through_unchanged():
    rule_alerts = [{"component": "cooler", "severity": "WARNING", "message": "r"}]
    fused = rules.fuse_alerts(rule_alerts, [])
    assert fused == rule_alerts
