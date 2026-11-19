"""Correctness tests for ml39's real Integrated Gradients port — exercises the actual plugin
(not FakePlugin) against real golden cases from inbox/a39/manifest.yaml.

Runs in its own subprocess (see _ml39_subprocess_check.py): ml1's model_loader (and a37's, once
merged) sets KERAS_BACKEND=torch at import time, which poisons ``tf.keras`` process-wide for
any plugin that needs real TensorFlow gradients, like this one — see that helper's docstring
and the manifest's known_issues. This only matters for a shared process (this test suite, the
local multi-model dev server); each model is its own pod in production.

Skipped automatically if the real artifacts aren't available locally (e.g. a bare checkout
without artifacts/ml39_wine_predictive_maintenance_xai/ populated).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = _REPO_ROOT / "artifacts" / "ml39_wine_predictive_maintenance_xai"
RAW_TENSOR_PATH = _REPO_ROOT / "inbox" / "a37" / "codigo" / "data" / "processed" / "tensors" / "X.npy"

pytestmark = pytest.mark.skipif(
    not (ARTIFACT_DIR.is_dir() and RAW_TENSOR_PATH.is_file()),
    reason="real ml39 artifacts / a37 raw tensor not available locally",
)

# Real golden cases from inbox/a39/manifest.yaml (steps=50, target_class_idx=None).
_GOLDEN_CASES = {
    33: {
        "predicted_class": {"cooler": 3, "valve": 100, "pump": 0, "accumulator": 130},
        "ig_top3": {
            "cooler": {"PS6": 0.000318, "TS4": 0.000241, "TS1": 0.000224},
            "valve": {"PS6": 0.000153, "PS5": 0.00014, "TS4": 0.000138},
            "pump": {"FS1": 0.000153, "EPS1": 0.000095, "VS1": 0.00009},
            "accumulator": {"PS6": 0.000229, "TS4": 0.000171, "FS1": 0.000168},
        },
    },
    220: {
        "predicted_class": {"cooler": 3, "valve": 100, "pump": 1, "accumulator": 130},
        "ig_top3": {
            "cooler": {"PS6": 0.000335, "TS4": 0.000298, "TS1": 0.000233},
            "valve": {"TS3": 0.000123, "TS4": 0.000112, "TS1": 0.000107},
            "pump": {"TS1": 0.00024, "FS1": 0.000207, "PS6": 0.000154},
            "accumulator": {"FS2": 0.000131, "TS1": 0.00012, "TS4": 0.000116},
        },
    },
    1842: {
        "predicted_class": {"cooler": 100, "valve": 90, "pump": 1, "accumulator": 100},
        "ig_top3": {
            "cooler": {"PS4": 0.003605, "TS1": 0.003385, "PS6": 0.002427},
            "valve": {"PS4": 0.00044, "TS1": 0.000301, "PS6": 0.000227},
            "pump": {"PS4": 0.000707, "TS1": 0.000516, "TS4": 0.000403},
            "accumulator": {"PS4": 0.000301, "TS1": 0.000189, "TS2": 0.000166},
        },
    },
}


@pytest.fixture(scope="module")
def golden_results() -> dict:
    """Run the real plugin against every golden cycle in a single isolated subprocess."""
    script = _REPO_ROOT / "tests" / "unit" / "_ml39_subprocess_check.py"
    cycle_ids = sorted(_GOLDEN_CASES)
    # ml1's model_loader (and a37's) mutates the real os.environ with KERAS_BACKEND=torch on
    # import — strip it so the child starts clean regardless of what already ran in this
    # pytest process (see this test file's and _ml39_subprocess_check.py's docstrings).
    clean_env = {k: v for k, v in os.environ.items() if k != "KERAS_BACKEND"}
    proc = subprocess.run(
        [sys.executable, str(script), json.dumps(cycle_ids)],
        cwd=_REPO_ROOT, env=clean_env, capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return {
        row["cycle_id"]: row
        for row in (json.loads(line) for line in proc.stdout.splitlines() if line.strip())
    }


@pytest.mark.parametrize("cycle_id", sorted(_GOLDEN_CASES))
def test_ig_reproduces_golden_case(golden_results, cycle_id):
    """Real Integrated Gradients top-3 sensors and |IG| values match the manifest's golden
    case within 1% relative tolerance (deterministic: no randomness in IG given steps=50)."""
    case = _GOLDEN_CASES[cycle_id]
    got_row = golden_results[cycle_id]

    assert got_row["predicted_class"] == case["predicted_class"]

    for head, expected_top3 in case["ig_top3"].items():
        got = got_row["ig_importance"][head]
        got_top3 = dict(sorted(got.items(), key=lambda kv: -kv[1])[:3])
        assert list(got_top3.keys()) == list(expected_top3.keys()), (
            f"head={head} cycle={cycle_id}: ranking mismatch {got_top3} vs {expected_top3}"
        )
        for sensor, expected_value in expected_top3.items():
            assert got_top3[sensor] == pytest.approx(expected_value, rel=0.01), (
                f"head={head} sensor={sensor} cycle={cycle_id}"
            )
