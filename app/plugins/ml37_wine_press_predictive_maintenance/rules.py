"""Neurosymbolic rule fusion for ml37 — faithful port of
src/neurosymbolic/symbolic_engine.py (check_rule_evidence, check_model, fuse_alerts).

See inbox/a37/manifest.yaml known_issues for a real quirk replicated on purpose here: the
"legacy" fused view (alerts, below) collapses multiple rule alerts of the same component with a
plain dict (last rule evaluated wins), which can silently drop a more severe alert if a less
severe one for the same component is evaluated afterwards (this only affects `accumulator`,
whose PS1_mean and PS1_std checks are independent `if` blocks, not elif). The delivered code is
aware of this (`would_be_overwritten_in_legacy` in its own evidence detail) — it is not fixed
here, only exposed: `rule_evidence` below carries every rule that fired, not just the legacy one.
"""
from __future__ import annotations

from app.plugins.ml37_wine_press_predictive_maintenance.constants import (
    MODEL_SUPPRESS_PROBA,
    MODEL_TRUST,
    RULE_BASELINE,
    SEVERITY_RANK,
    TRUST_THRESHOLD,
)

# component -> {predicted_class: severity} — only classes below the nominal/OK one alert.
# Faithful to symbolic_engine.py::CRITICAL_CLASSES.
_CRITICAL_CLASSES = {
    "cooler": {3: "CRITICAL", 20: "WARNING"},
    "valve": {73: "CRITICAL", 80: "CRITICAL", 90: "WARNING"},
    "pump": {1: "WARNING", 2: "CRITICAL"},
    "accumulator": {90: "CRITICAL", 100: "CRITICAL", 115: "WARNING"},
}
_DOWNGRADE = {"CRITICAL": "WARNING", "WARNING": "INFO"}


def _alert(component: str, severity: str, msg: str, value: float, *, rule_name: str) -> dict:
    return {
        "component": component, "source": "rule", "severity": severity,
        "message": msg, "value": value, "rule_name": rule_name,
    }


def _check_cooler_rule(stats: dict[str, float]) -> list[dict]:
    ts1 = stats.get("TS1_mean")
    if ts1 is None:
        return []
    if ts1 > RULE_BASELINE["TS1_mean_crit"]:
        return [_alert("cooler", "CRITICAL", f"TS1_mean={ts1:.2f} supera umbral crítico "
                       f"({RULE_BASELINE['TS1_mean_crit']})", ts1, rule_name="ts1_mean_critical")]
    if ts1 > RULE_BASELINE["TS1_mean_max"]:
        return [_alert("cooler", "WARNING", f"TS1_mean={ts1:.2f} elevada", ts1,
                       rule_name="ts1_mean_warning")]
    return []


def _check_valve_rule(stats: dict[str, float]) -> list[dict]:
    ps3, ps4 = stats.get("PS3_mean"), stats.get("PS4_mean")
    if ps3 is None or ps4 is None:
        return []
    dp = ps3 - ps4
    if dp < RULE_BASELINE["dp_valve_min_crit"]:
        return [_alert("valve", "CRITICAL", f"Diferencial PS3-PS4={dp:.3f} bar (fallo de "
                       "válvula)", dp, rule_name="dp_valve_critical")]
    if dp < RULE_BASELINE["dp_valve_min_warn"]:
        return [_alert("valve", "WARNING", f"Diferencial PS3-PS4={dp:.3f} bar bajo", dp,
                       rule_name="dp_valve_warning")]
    return []


def _check_pump_rule(stats: dict[str, float]) -> list[dict]:
    fs1, eps1 = stats.get("FS1_mean"), stats.get("EPS1_mean")
    if fs1 is None or not eps1 or eps1 <= 0:
        return []
    eff = fs1 / eps1
    if eff < RULE_BASELINE["pump_eff_min_crit"]:
        return [_alert("pump", "CRITICAL", f"Eficiencia bomba={eff:.5f} (fuga severa)", eff,
                       rule_name="pump_eff_critical")]
    if eff < RULE_BASELINE["pump_eff_min_warn"]:
        return [_alert("pump", "WARNING", f"Eficiencia bomba={eff:.5f} baja", eff,
                       rule_name="pump_eff_warning")]
    return []


def _check_accumulator_rule(stats: dict[str, float]) -> list[dict]:
    """PS1_mean and PS1_std are two INDEPENDENT checks (not elif): both can fire together —
    the source of the legacy-view collapse quirk (see module docstring)."""
    alerts: list[dict] = []
    ps1_mean, ps1_std = stats.get("PS1_mean"), stats.get("PS1_std")
    if ps1_mean is not None:
        if ps1_mean < RULE_BASELINE["PS1_mean_min_crit"]:
            msg = f"PS1_mean={ps1_mean:.2f} bar crítico"
            alerts.append(
                _alert("accumulator", "CRITICAL", msg, ps1_mean, rule_name="ps1_mean_critical")
            )
        elif ps1_mean < RULE_BASELINE["PS1_mean_min_warn"]:
            msg = f"PS1_mean={ps1_mean:.2f} bar bajo"
            alerts.append(
                _alert("accumulator", "WARNING", msg, ps1_mean, rule_name="ps1_mean_warning")
            )
    if ps1_std is not None and ps1_std > RULE_BASELINE["PS1_std_max_warn"]:
        msg = f"PS1_std={ps1_std:.3f} (presión inestable)"
        alerts.append(
            _alert("accumulator", "WARNING", msg, ps1_std, rule_name="ps1_std_warning")
        )
    return alerts


def check_rule_evidence(stats: dict[str, float]) -> list[dict]:
    """Every rule that fires for this cycle (0 to several per component) — faithful port of
    symbolic_engine.py::check_rule_evidence, same thresholds and same evaluation order."""
    return [
        *_check_cooler_rule(stats), *_check_valve_rule(stats),
        *_check_pump_rule(stats), *_check_accumulator_rule(stats),
    ]


def _legacy_rule_view(rule_alerts: list[dict]) -> list[dict]:
    """Collapse to one alert per component — last rule evaluated wins (see module docstring:
    this is a real, deliberately-replicated quirk of the delivered code for `accumulator`)."""
    by_component: dict[str, dict] = {}
    for alert in rule_alerts:
        by_component[alert["component"]] = alert
    return list(by_component.values())


def check_model(predictions: dict[str, dict]) -> list[dict]:
    """Model-side alerts with MODEL_TRUST downgrade/suppression — faithful port of
    symbolic_engine.py::check_model."""
    alerts: list[dict] = []
    for head, pred in predictions.items():
        cls = pred["class"]
        proba = pred.get("proba", 1.0)
        severity = _CRITICAL_CLASSES.get(head, {}).get(cls)
        if severity is None:
            continue
        trust = MODEL_TRUST.get(head, 0.5)
        if trust < TRUST_THRESHOLD and proba < MODEL_SUPPRESS_PROBA:
            continue
        if trust < TRUST_THRESHOLD:
            severity = _DOWNGRADE.get(severity, "INFO")
        alerts.append({
            "component": head, "source": "model", "severity": severity,
            "message": f"Clase predicha={cls} (proba={proba:.2f}, trust={trust:.0%})",
            "value": proba,
        })
    return alerts


def fuse_alerts(rule_alerts: list[dict], model_alerts: list[dict]) -> list[dict]:
    """Faithful port of symbolic_engine.py::fuse_alerts — agreement between rule and model for
    the same component escalates severity one rank above whichever alone was worse (capped at
    EMERGENCY), never a simple max()."""
    by_rule = {a["component"]: a for a in rule_alerts}
    by_model = {a["component"]: a for a in model_alerts}
    components = set(by_rule) | set(by_model)

    fused: list[dict] = []
    for component in components:
        r, m = by_rule.get(component), by_model.get(component)
        if r and m:
            max_rank = max(SEVERITY_RANK[r["severity"]], SEVERITY_RANK[m["severity"]])
            new_rank = min(max_rank + 1, SEVERITY_RANK["EMERGENCY"])
            new_severity = next(k for k, v in SEVERITY_RANK.items() if v == new_rank)
            fused.append({
                "component": component, "source": "combined", "severity": new_severity,
                "message": f"[RULE] {r['message']} | [MODEL] {m['message']}",
            })
        else:
            fused.append(r or m)

    return sorted(fused, key=lambda a: -SEVERITY_RANK[a["severity"]])


def evaluate_cycle_alerts(stats: dict[str, float], predictions: dict[str, dict]) -> dict:
    """Full neurosymbolic output for one cycle: the legacy fused view (`alerts`, what
    alerts_all.parquet carries) plus the complete rule evidence (`rule_evidence`, every rule
    that fired — including any the legacy view's component-collapse would have dropped)."""
    rule_evidence = check_rule_evidence(stats)
    legacy_rules = _legacy_rule_view(rule_evidence)
    model_alerts = check_model(predictions)
    fused = fuse_alerts(legacy_rules, model_alerts)
    return {"alerts": fused, "rule_evidence": rule_evidence}
