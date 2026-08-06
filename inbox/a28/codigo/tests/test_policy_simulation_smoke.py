from __future__ import annotations

from src.utils import read_json
from tests.conftest import REPO_ROOT, ensure_repro_smoke_pipeline


def test_policy_simulation_smoke_generates_guardrailed_metrics() -> None:
    ensure_repro_smoke_pipeline()
    summary_path = REPO_ROOT / "models/metrics/summary/policy_simulation_latest__mixed_context.json"
    summary = read_json(summary_path)

    assert summary["trigger_rule_respected"] is True
    assert summary["stockout_guardrail_pass"] is True
    assert "aggregate_excess_reduction_pct" in summary
