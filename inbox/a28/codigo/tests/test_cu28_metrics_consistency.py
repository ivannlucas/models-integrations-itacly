from __future__ import annotations

import math
import re

import pandas as pd
import pytest

from scripts import audit_doc_metrics_alignment
from src.data_processing.synthetic_plant import _merge_procurement_target_parameters
from src.evaluation.metrics import (
    compute_policy_metrics,
    compute_regression_metrics,
    compute_trigger_metrics,
)
from src.reproducibility import SMOKE_MODE, build_reproducibility_config
from src.reproducibility.mixed_context import QUANTITY_TARGET, quantity_feature_columns
from src.reproducibility.runtime import official_paths, require_official_end_to_end_run
from src.utils import read_json
from tests.conftest import REPO_ROOT, ensure_repro_smoke_pipeline


def _assert_close(left: float, right: float) -> None:
    assert math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)


def _config_and_paths() -> tuple[dict, dict]:
    config = build_reproducibility_config("config/config.yaml", mode=SMOKE_MODE)
    return config, official_paths(config)


def test_trigger_confusion_matrix_matches_predictions_artifact_and_report() -> None:
    ensure_repro_smoke_pipeline()
    _, paths = _config_and_paths()
    artifact = read_json(paths["trigger_metrics_json"])
    predictions = pd.read_csv(REPO_ROOT / artifact["prediction_paths"]["train"])
    recalculated = compute_trigger_metrics(
        predictions["actual"],
        predictions["prediction"],
        predictions["probability_buy"],
        split="train",
    )

    assert recalculated["confusion_matrix"] == artifact["train"]["confusion_matrix"]
    report_line = next(
        line
        for line in paths["official_metrics_report"].read_text(encoding="utf-8").splitlines()
        if line.startswith("| train |")
    )
    confusion = artifact["train"]["confusion_matrix"]
    assert report_line.endswith(
        f"| {confusion['true_negative']} | {confusion['false_positive']} | "
        f"{confusion['false_negative']} | {confusion['true_positive']} |"
    )


def test_upstream_regression_metrics_match_generated_json() -> None:
    ensure_repro_smoke_pipeline()
    _, paths = _config_and_paths()
    summary = read_json(paths["baseline_summary_json"])
    record = next(
        run
        for run in summary["runs"]
        if run["model_family"] == "linear_regression"
        and run["feature_set"] == "ablation_reduced_context"
    )
    payload = read_json(REPO_ROOT / record["metrics_path"])

    for split in ["validation", "test"]:
        predictions = pd.read_csv(REPO_ROOT / payload["prediction_paths"][split])
        metrics = compute_regression_metrics(
            predictions["actual"],
            predictions["prediction"],
            split=split,
            model=payload["model_name"],
            target=payload["target_column"],
        )
        for field in ["n_samples", "mae", "rmse", "r2"]:
            _assert_close(metrics[field], payload["metrics"][split][field])


def test_quantity_optimizer_metrics_match_official_artifact() -> None:
    ensure_repro_smoke_pipeline()
    _, paths = _config_and_paths()
    artifact = read_json(paths["quantity_optimizer_metrics_json"])
    predictions = pd.read_csv(REPO_ROOT / artifact["prediction_paths"]["test"])
    metrics = compute_regression_metrics(
        predictions["actual"],
        predictions["prediction"],
        split="test",
        model=artifact["model_family"],
        target=artifact["target_column"],
    )

    for field in ["n_samples", "mae", "rmse", "r2"]:
        _assert_close(metrics[field], artifact["test"][field])


def test_quantity_optimizer_supervised_baseline_comparison_artifact() -> None:
    ensure_repro_smoke_pipeline()
    _, paths = _config_and_paths()
    artifact = read_json(paths["quantity_optimizer_baseline_comparison_json"])

    assert paths["quantity_optimizer_baseline_comparison_csv"].is_file()
    assert artifact["component"] == "quantity_optimizer"
    assert artifact["target"] == QUANTITY_TARGET
    assert artifact["evaluation_filter"] == "purchase_trigger_label == 1"
    assert artifact["baseline"] == "DummyRegressor(strategy='mean')"
    assert artifact["official_model"] == "Ridge"
    assert QUANTITY_TARGET not in artifact["feature_columns"]
    assert "purchase_trigger_label" not in artifact["feature_columns"]

    metrics = artifact["metrics"]
    expected_pairs = {
        (model, split)
        for split in ["train", "validation", "test"]
        for model in ["DummyRegressor", "Ridge"]
    }
    assert {(row["model"], row["split"]) for row in metrics} == expected_pairs
    for row in metrics:
        assert {"rmse", "mae", "r2", "n_rows"}.issubset(row)
        assert row["n_rows"] > 0


def test_quantity_optimizer_supervised_baseline_comparison_in_summary() -> None:
    ensure_repro_smoke_pipeline()
    _, paths = _config_and_paths()
    summary = read_json(paths["metrics_summary_json"])

    comparison = summary["quantity_optimizer_baseline_comparison"]
    assert comparison["target"] == QUANTITY_TARGET
    assert comparison["evaluation_filter"] == "purchase_trigger_label == 1"
    assert {row["model"] for row in comparison["metrics"]} == {"DummyRegressor", "Ridge"}


def test_doc_metrics_alignment_report_is_generated_from_official_artifacts() -> None:
    ensure_repro_smoke_pipeline()

    result = audit_doc_metrics_alignment.main(["--scope", "mixed_context", "--fail-on-mismatch"])
    payload = read_json(REPO_ROOT / "reports/audit/cu28_doc_metrics_alignment.json")

    assert result == 0
    assert payload["status"] == "PASS"
    assert payload["checks_passed"] == payload["checks_total"]
    assert payload["documentation_scan"]["obsolete_findings"] == []
    expected_sections = {
        "upstream_predictor",
        "purchase_trigger",
        "quantity_optimizer",
        "quantity_optimizer_baseline_comparison",
        "policy_simulation",
        "synthetic_procurement_need_formula",
    }
    assert set(payload["documentable_metrics"]) == expected_sections
    comparison_models = {
        row["model"]
        for row in payload["documentable_metrics"]["quantity_optimizer_baseline_comparison"]
    }
    assert comparison_models == {"DummyRegressor", "Ridge"}


def test_policy_excess_matches_period_csv_and_summary_json() -> None:
    ensure_repro_smoke_pipeline()
    config, paths = _config_and_paths()
    summary = read_json(paths["policy_simulation_summary_json"])
    period_df = pd.read_csv(REPO_ROOT / summary["period_metrics_csv_path"])
    recalculated = compute_policy_metrics(
        period_df,
        guardrail_name="bounded_stockout_increase",
        allowed_stockout_increase_pct=float(
            config["policy_simulation"]["kpi_definition"]["allowed_stockout_increase_pct"]
        ),
    )

    for field in [
        "baseline_excess_tons",
        "policy_excess_tons",
        "absolute_excess_reduction_tons",
        "aggregate_excess_reduction_pct",
        "baseline_stockout_tons",
        "policy_stockout_tons",
        "n_periods",
    ]:
        _assert_close(recalculated[field], summary[field])


def test_synthetic_procurement_weights_used_by_code_match_config() -> None:
    config, _ = _config_and_paths()
    configured = config["synthetic_data"]["simulation_parameters"][
        "procurement_target_parameters"
    ]
    used = _merge_procurement_target_parameters(
        config["synthetic_data"]["simulation_parameters"]
    )

    assert used["pressure_snapshot_blend_weight"] == configured[
        "pressure_snapshot_blend_weight"
    ]
    assert used["pressure_coverage_gap_blend_weight"] == configured[
        "pressure_coverage_gap_blend_weight"
    ]


def test_generated_formula_documentation_matches_config() -> None:
    ensure_repro_smoke_pipeline()
    config, paths = _config_and_paths()
    generated = read_json(paths["official_formula_json"])
    configured = config["synthetic_data"]["simulation_parameters"][
        "procurement_target_parameters"
    ]

    assert generated["pressure_snapshot_blend_weight"] == configured[
        "pressure_snapshot_blend_weight"
    ]
    assert generated["pressure_coverage_gap_blend_weight"] == configured[
        "pressure_coverage_gap_blend_weight"
    ]


def test_old_metric_and_formula_values_are_not_current_results() -> None:
    ensure_repro_smoke_pipeline()
    files = [
        *sorted((REPO_ROOT / "docs").rglob("*.md")),
        *sorted((REPO_ROOT / "reports" / "official").rglob("*.md")),
    ]
    patterns = [
        re.compile(r"(77[,.]24|98[,.]80|129[,.]43|208[,.]15)"),
        re.compile(
            r"(true_negative[^\n]{0,40}366|false_positive[^\n]{0,40}5\b|"
            r"false_negative[^\n]{0,40}36\b|true_positive[^\n]{0,40}407\b)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(pressure_snapshot_blend_weight[^\n]{0,40}0[,.]22|"
            r"pressure_coverage_gap_blend_weight[^\n]{0,40}0[,.]18)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(reduction|reducci[oó]n|excess|excedente)[^\n]{0,80}"
            r"(21[,.]5\b|37686|37\.686|29569|29\.569)",
            re.IGNORECASE,
        ),
    ]
    findings = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        if any(pattern.search(text) for pattern in patterns):
            findings.append(path.relative_to(REPO_ROOT).as_posix())

    assert findings == []


def test_quantity_optimizer_does_not_use_ground_truth_trigger_as_feature() -> None:
    frame = pd.DataFrame(
        columns=[
            "purchase_trigger_label",
            "purchase_trigger_proba_heuristic",
            "current_inventory_tons",
            "expected_requirement_tons",
        ]
    )
    assert "purchase_trigger_label" not in quantity_feature_columns(frame)


def test_partial_run_cannot_publish_latest_artifacts() -> None:
    config, _ = _config_and_paths()
    with pytest.raises(RuntimeError):
        require_official_end_to_end_run(config, stage="train")
