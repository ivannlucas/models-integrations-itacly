"""Vendored (trimmed) from inbox/a46/codigo/.../src/training/artifact_validation.py.

Kept: validate_feature_artifacts, validate_policy_artifact — pure structural checks on
already-loaded artifacts (no external file paths involved), so they work unchanged
regardless of this plugin's S3-flattened artifact layout. Dropped: dataframe/file
fingerprinting and the byte/canonical-hash cross-file matching (validate_manifest_contract
in the original training/artifact_contract.py) — those assume the delivery repo's own
directory layout and TrainConfig field set, and would require mirroring its evolving
dataclasses field-for-field forever just to keep hashes matching. The checkpoint/
architecture shape check in model_arch.py::validate_checkpoint_compatibility already
catches the practically important failure mode (wrong artifact bundle uploaded).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, Sequence


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def payload_sha256(payload: Any) -> str:
    """Stable content hash of a JSON-serializable payload (dict key order independent)."""
    raw = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def feature_names_hash(feature_names: Sequence[str]) -> str:
    return payload_sha256([str(name) for name in feature_names])


def validate_feature_artifacts(artifacts: Any, scenario: str, feature_names: Sequence[str]) -> dict[str, Any]:
    """Structural sanity check on a loaded FeatureArtifacts bundle — raises ValueError on failure."""
    errors: list[str] = []
    numeric_names = list(getattr(artifacts, "numeric_feature_names", []) or [])
    medians = getattr(artifacts, "medians", {}) or {}
    iqrs = getattr(artifacts, "iqrs", {}) or {}
    full_names = list(getattr(artifacts, "full_feature_names", []) or [])
    no_clock_names = list(getattr(artifacts, "no_clock_feature_names", []) or [])
    selected_names = list(feature_names)

    if not numeric_names:
        errors.append("feature_artifacts.numeric_feature_names is empty.")
    missing_medians = [name for name in numeric_names if name not in medians]
    missing_iqrs = [name for name in numeric_names if name not in iqrs]
    bad_iqrs = [
        name for name in numeric_names
        if name in iqrs and (not math.isfinite(float(iqrs[name])) or float(iqrs[name]) <= 0)
    ]
    if missing_medians:
        errors.append(f"Missing medians for numeric features: {missing_medians[:10]}")
    if missing_iqrs:
        errors.append(f"Missing IQRs for numeric features: {missing_iqrs[:10]}")
    if bad_iqrs:
        errors.append(f"Invalid non-positive/non-finite IQRs: {bad_iqrs[:10]}")
    if not full_names:
        errors.append("feature_artifacts.full_feature_names is empty.")
    if scenario == "no_clock" and not no_clock_names:
        errors.append("feature_artifacts.no_clock_feature_names is empty for scenario no_clock.")
    if len(set(selected_names)) != len(selected_names):
        errors.append("Selected feature list contains duplicated names.")
    if scenario not in {"full", "no_clock"}:
        errors.append(f"Unknown scenario '{scenario}'. Expected 'full' or 'no_clock'.")
    if selected_names and full_names and not set(selected_names).issubset(set(full_names)):
        missing_from_full = sorted(set(selected_names) - set(full_names))
        errors.append(f"Selected features are not a subset of full features: {missing_from_full[:10]}")

    report = {
        "scenario": scenario,
        "n_numeric_features": len(numeric_names),
        "n_selected_features": len(selected_names),
        "selected_feature_names_hash": feature_names_hash(selected_names),
        "errors": errors,
        "ok": not errors,
    }
    if errors:
        raise ValueError("Feature artifact compatibility check failed: " + "; ".join(errors))
    return report


def validate_policy_artifact(policy: Mapping[str, Any], scenario: str) -> dict[str, Any]:
    """Structural sanity check on a loaded alert-policy dict — raises ValueError on failure."""
    required = {
        "clog_prob_thr",
        "watch_foul_prob_thr",
        "actionable_foul_prob_thr",
        "tau_clog",
        "tau_foul_watch",
        "tau_unplanned",
        "severity_incipient_thr",
        "severity_advanced_thr",
        "cooldown_min",
    }
    missing = sorted(required - set(policy.keys()))
    if missing:
        raise ValueError(f"Policy thresholds for scenario '{scenario}' are incomplete. Missing keys: {missing}")
    return {
        "scenario": scenario,
        "policy_hash": payload_sha256(dict(policy)),
        "required_keys": sorted(required),
        "ok": True,
    }
