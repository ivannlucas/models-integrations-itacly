from __future__ import annotations

from src.utils import read_json
from tests.conftest import REPO_ROOT, ensure_repro_smoke_pipeline


def test_train_smoke_generates_official_artifacts() -> None:
    ensure_repro_smoke_pipeline()

    artifact_paths = [
        REPO_ROOT / "models/artifacts/upstream_predictor_latest__mixed_context.pkl",
        REPO_ROOT / "models/artifacts/purchase_trigger_latest__mixed_context.pkl",
        REPO_ROOT / "models/artifacts/quantity_optimizer_latest__mixed_context.pkl",
        REPO_ROOT / "models/artifacts/model_manifest__mixed_context.json",
    ]
    for path in artifact_paths:
        assert path.exists(), path
        assert path.stat().st_size > 0

    baseline_summary = read_json(REPO_ROOT / "models/metrics/summary/baseline_comparison_latest__mixed_context.json")
    assert baseline_summary["selection_policy"]["primary_metric"] == "validation_rmse"
    assert baseline_summary["best_baseline_run"]["target_column"] == "synthetic_procurement_need"
