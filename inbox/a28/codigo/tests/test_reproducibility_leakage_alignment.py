from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.evaluation import pipeline as evaluation_pipeline
from src.reproducibility.mixed_context import (
    PROHIBITED_UPSTREAM_INPUTS,
    quantity_feature_columns,
    validate_feature_columns_for_stage,
)
from src.utils import read_json
from tests.conftest import REPO_ROOT, ensure_repro_smoke_pipeline


def test_mixed_context_reference_window() -> None:
    ensure_repro_smoke_pipeline()
    metadata = read_json(REPO_ROOT / "data/splits/baseline/default__mixed_context/split_metadata.json")
    splits = metadata["splits"]

    assert sum(split["rows"] for split in splits.values()) == 1161
    assert splits["train"]["rows"] == 812
    assert splits["validation"]["rows"] == 174
    assert splits["test"]["rows"] == 175
    assert splits["test"]["date_end"] == "2026-05-18"


def test_quantity_optimizer_no_ground_truth_trigger_feature() -> None:
    frame = pd.DataFrame(
        columns=[
            "purchase_trigger_label",
            "purchase_trigger_proba_heuristic",
            "current_inventory_tons",
            "expected_requirement_tons",
            "lead_time_days",
            "safety_coverage_days",
            "expected_yield_rate",
            "expected_waste_rate",
            "replenishment_gap_tons",
            "demand_index",
            "supply_index",
            "purchase_price_index",
            "demand_supply_gap",
        ]
    )

    feature_columns = quantity_feature_columns(frame)

    assert "purchase_trigger_label" not in feature_columns
    validate_feature_columns_for_stage(feature_columns, stage="quantity_optimizer")
    with pytest.raises(ValueError):
        validate_feature_columns_for_stage([*feature_columns, "purchase_trigger_label"], stage="quantity_optimizer")


def test_downstream_features_excluded_from_upstream() -> None:
    safe_features = ["demand_index", "supply_index", "purchase_price_index"]

    validate_feature_columns_for_stage(safe_features, stage="upstream")
    for prohibited in PROHIBITED_UPSTREAM_INPUTS:
        with pytest.raises(ValueError):
            validate_feature_columns_for_stage([*safe_features, prohibited], stage="upstream")


def test_gating_forces_zero_quantity() -> None:
    ensure_repro_smoke_pipeline()
    predictions = pd.read_csv(REPO_ROOT / "data/predictions/predictions_latest__mixed_context.csv")
    blocked = predictions[pd.to_numeric(predictions["purchase_trigger_flag"], errors="coerce").fillna(0).astype(int).eq(0)]

    assert pd.to_numeric(blocked["order_quantity_tons"], errors="coerce").fillna(0.0).eq(0.0).all()


def test_supply_index_no_backfill_before_first_observation() -> None:
    ensure_repro_smoke_pipeline()
    context = pd.read_csv(REPO_ROOT / "data/processed/external/context/context_weekly_for_simulation.csv")
    context["date"] = pd.to_datetime(context["date"], errors="coerce")
    pre_2021 = context[context["date"] < pd.Timestamp("2021-01-01")]
    first_valid_supply_date = context.loc[context["supply_index"].notna(), "date"].min()

    assert pre_2021["supply_index"].isna().all()
    assert first_valid_supply_date >= pd.Timestamp("2021-01-01")


def test_get_stats_does_not_overwrite_full_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    official_manifest = tmp_path / "reproducibility_manifest__mixed_context.json"
    calls: dict[str, object] = {}

    def fake_official_paths(config: dict) -> dict[str, Path]:
        return {
            "metrics_summary_json": tmp_path / "metrics_summary__mixed_context.json",
            "metrics_summary_csv": tmp_path / "metrics_summary__mixed_context.csv",
            "repro_manifest": official_manifest,
        }

    def fake_build_manifest(config: dict, **kwargs: object) -> dict[str, object]:
        calls.update(kwargs)
        return {"scope": "mixed_context", "manifest_scope": kwargs.get("manifest_scope")}

    class Logger:
        def info(self, *_args: object, **_kwargs: object) -> None:
            return None

    monkeypatch.setattr(evaluation_pipeline, "build_metrics_summary", lambda _config: {"ok": True})
    monkeypatch.setattr(evaluation_pipeline, "official_paths", fake_official_paths)
    monkeypatch.setattr(evaluation_pipeline, "build_reproducibility_manifest", fake_build_manifest)

    result = evaluation_pipeline.run_reproducibility_get_stats({"project": {"repo_root": str(tmp_path)}}, Logger())

    assert Path(calls["output_path"]).name == "reproducibility_manifest_partial__mixed_context.json"
    assert Path(calls["output_path"]) != official_manifest
    assert calls["manifest_scope"] == "partial_get_stats"
    assert result["manifest_scope"] == "partial_get_stats"
