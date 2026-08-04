from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_processing.synthetic_plant import _merge_procurement_target_parameters
from src.evaluation.metrics import (
    compute_policy_metrics,
    compute_regression_metrics,
    compute_trigger_metrics,
)
from src.reproducibility import SMOKE_MODE, build_reproducibility_config
from src.reproducibility.runtime import official_paths
from src.utils import read_json, write_json

REPORT_JSON = REPO_ROOT / "reports" / "audit" / "cu28_doc_metrics_alignment.json"
REPORT_MD = REPO_ROOT / "reports" / "audit" / "cu28_doc_metrics_alignment.md"
SCOPE = "mixed_context"

DOC_SCAN_PATHS = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "DELIVERY_README_CU28.md",
    REPO_ROOT / "docs",
    REPO_ROOT / "reports" / "official",
]

OBSOLETE_PATTERNS = {
    "legacy_upstream_rmse": re.compile(r"(77[,.]24|98[,.]80|129[,.]43|208[,.]15)"),
    "legacy_trigger_train_matrix": re.compile(
        r"(true_negative[^\n]{0,50}366|false_positive[^\n]{0,50}5\b|"
        r"false_negative[^\n]{0,50}36\b|true_positive[^\n]{0,50}407\b)",
        re.IGNORECASE,
    ),
    "legacy_formula_weights": re.compile(
        r"(pressure_snapshot_blend_weight[^\n]{0,50}0[,.]22|"
        r"pressure_coverage_gap_blend_weight[^\n]{0,50}0[,.]18)",
        re.IGNORECASE,
    ),
    "legacy_policy_values": re.compile(
        r"(reduction|reducci[oÃ³]n|excess|excedente)[^\n]{0,80}"
        r"(21[,.]5\b|37686|37\.686|29569|29\.569)",
        re.IGNORECASE,
    ),
}


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def _relative(path: str | Path) -> str:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _same_number(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    except (TypeError, ValueError):
        return left == right


def _add_check(checks: list[Check], name: str, passed: bool, detail: str) -> None:
    checks.append(Check(name=name, passed=passed, detail=detail))


def _compare_fields(
    checks: list[Check],
    *,
    name: str,
    expected: dict[str, Any],
    actual: dict[str, Any],
    fields: list[str],
) -> None:
    mismatches = [
        f"{field}: expected={expected.get(field)!r} actual={actual.get(field)!r}"
        for field in fields
        if not _same_number(expected.get(field), actual.get(field))
    ]
    _add_check(checks, name, not mismatches, "match" if not mismatches else "; ".join(mismatches))


def _find_run(summary: dict[str, Any], model_family: str, feature_set: str) -> dict[str, Any]:
    for record in summary.get("runs", []):
        if record.get("model_family") == model_family and record.get("feature_set") == feature_set:
            return dict(record)
    raise KeyError(f"Missing run model_family={model_family} feature_set={feature_set}")


def _read_relative_csv(path_value: str | Path) -> pd.DataFrame:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return pd.read_csv(path)


def _metric_record(
    *,
    section: str,
    source: str,
    values: dict[str, Any],
) -> dict[str, Any]:
    return {
        "section": section,
        "source": source,
        **values,
    }


def _collect_upstream(
    checks: list[Check],
    baseline: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary_source = "models/metrics/summary/baseline_comparison_latest__mixed_context.json"
    for model_family in ["linear_regression", "neuroevolution"]:
        record = _find_run(baseline, model_family, "ablation_reduced_context")
        run_payload = read_json(REPO_ROOT / record["metrics_path"])
        for split in ["validation", "test"]:
            predictions = _read_relative_csv(run_payload["prediction_paths"][split])
            recalculated = compute_regression_metrics(
                predictions["actual"],
                predictions["prediction"],
                split=split,
                model=run_payload["model_name"],
                target=run_payload["target_column"],
                include_mape=True,
                include_distribution=True,
            )
            _compare_fields(
                checks,
                name=f"upstream {model_family} {split}: predictions vs run JSON",
                expected=run_payload["metrics"][split],
                actual=recalculated,
                fields=["n_samples", "mae", "rmse", "r2", "mape", "prediction_bias"],
            )
            summary_metrics = {
                field: record[f"{split}_{field}"]
                for field in ["n_samples", "mae", "rmse", "r2", "mape", "prediction_bias"]
            }
            _compare_fields(
                checks,
                name=f"upstream {model_family} {split}: run JSON vs summary",
                expected=run_payload["metrics"][split],
                actual=summary_metrics,
                fields=["n_samples", "mae", "rmse", "r2", "mape", "prediction_bias"],
            )
            rows.append(
                _metric_record(
                    section="upstream_predictor",
                    source=f"{summary_source}; {record['metrics_path']}",
                    values={
                        "model_family": model_family,
                        "model_name": run_payload["model_name"],
                        "feature_set": record["feature_set"],
                        "target": record["target_column"],
                        "split": split,
                        "n_samples": record[f"{split}_n_samples"],
                        "mae": record[f"{split}_mae"],
                        "rmse": record[f"{split}_rmse"],
                        "r2": record[f"{split}_r2"],
                        "mape": record[f"{split}_mape"],
                        "prediction_bias": record[f"{split}_prediction_bias"],
                    },
                )
            )
    return rows


def _collect_trigger(checks: list[Check], trigger: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source = "models/metrics/summary/trigger_metrics_latest__mixed_context.json"
    for split in ["train", "validation", "test"]:
        predictions = _read_relative_csv(trigger["prediction_paths"][split])
        recalculated = compute_trigger_metrics(
            predictions["actual"],
            predictions["prediction"],
            predictions["probability_buy"],
            split=split,
        )
        fields = [
            "n_samples",
            "accuracy",
            "balanced_accuracy",
            "recall_buy",
            "recall_do_not_buy",
            "f1_buy",
            "f1_do_not_buy",
            "false_negative_rate",
        ]
        _compare_fields(
            checks,
            name=f"purchase trigger {split}: predictions vs summary JSON",
            expected=trigger[split],
            actual=recalculated,
            fields=fields,
        )
        confusion = trigger[split]["confusion_matrix"]
        actual_buy = int(confusion["true_positive"] + confusion["false_negative"])
        actual_do_not_buy = int(confusion["true_negative"] + confusion["false_positive"])
        n_rows = int(trigger[split]["n_samples"])
        buy_pct = actual_buy / max(actual_buy + actual_do_not_buy, 1)
        _add_check(
            checks,
            f"purchase trigger {split}: confusion matrix sums to n",
            actual_buy + actual_do_not_buy == n_rows,
            f"actual_buy={actual_buy} actual_do_not_buy={actual_do_not_buy} n={n_rows}",
        )
        _add_check(
            checks,
            f"purchase trigger {split}: BUY pct from counts",
            _same_number(buy_pct, trigger[split]["positive_rate_actual"]),
            f"buy_pct={buy_pct} positive_rate_actual={trigger[split]['positive_rate_actual']}",
        )
        rows.append(
            _metric_record(
                section="purchase_trigger",
                source=f"{source}; {trigger['prediction_paths'][split]}",
                values={
                    "split": split,
                    "n_samples": n_rows,
                    "actual_buy": actual_buy,
                    "actual_do_not_buy": actual_do_not_buy,
                    "buy_pct": buy_pct,
                    "accuracy": trigger[split]["accuracy"],
                    "balanced_accuracy": trigger[split]["balanced_accuracy"],
                    "precision_buy": trigger[split]["precision_by_class"]["BUY"],
                    "precision_do_not_buy": trigger[split]["precision_by_class"]["DO_NOT_BUY"],
                    "recall_buy": trigger[split]["recall_buy"],
                    "recall_do_not_buy": trigger[split]["recall_do_not_buy"],
                    "f1_buy": trigger[split]["f1_buy"],
                    "f1_do_not_buy": trigger[split]["f1_do_not_buy"],
                    "false_negative_rate": trigger[split]["false_negative_rate"],
                    "true_negative": confusion["true_negative"],
                    "false_positive": confusion["false_positive"],
                    "false_negative": confusion["false_negative"],
                    "true_positive": confusion["true_positive"],
                },
            )
        )
    return rows


def _collect_quantity(
    checks: list[Check],
    quantity: dict[str, Any],
    quantity_dummy: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    dummy_rows: list[dict[str, Any]] = []
    source = "models/metrics/summary/quantity_optimizer_latest__mixed_context.json"
    prediction_frames = {
        split: _read_relative_csv(quantity["prediction_paths"][split])
        for split in ["train", "validation", "test"]
    }
    for split, predictions in prediction_frames.items():
        recalculated = compute_regression_metrics(
            predictions["actual"],
            predictions["prediction"],
            split=split,
            model=quantity["model_family"],
            target=quantity["target_column"],
            include_mape=True,
            include_distribution=True,
        )
        _compare_fields(
            checks,
            name=f"quantity optimizer {split}: predictions vs summary JSON",
            expected=quantity[split],
            actual=recalculated,
            fields=["n_samples", "mae", "rmse", "r2", "mape", "prediction_bias"],
        )
        baseline = compute_regression_metrics(
            predictions["actual"],
            predictions["baseline_prediction"],
            split=split,
            model=quantity["baseline_comparison"].get("column") or quantity["baseline_comparison"].get("name"),
            target=quantity["target_column"],
            include_mape=True,
            include_distribution=False,
        )
        _compare_fields(
            checks,
            name=f"quantity optimizer {split}: baseline_order_quantity_tons vs summary JSON",
            expected=quantity[split]["baseline_comparison"],
            actual=baseline,
            fields=["n_samples", "mae", "rmse", "r2", "mape"],
        )
        rows.append(
            _metric_record(
                section="quantity_optimizer",
                source=f"{source}; {quantity['prediction_paths'][split]}",
                values={
                    "model": "Ridge",
                    "split": split,
                    "target": quantity["target_column"],
                    "n_samples": quantity[split]["n_samples"],
                    "mae": quantity[split]["mae"],
                    "rmse": quantity[split]["rmse"],
                    "r2": quantity[split]["r2"],
                    "mape": quantity[split]["mape"],
                    "baseline_order_quantity_tons_mae": quantity[split]["baseline_comparison"]["mae"],
                    "baseline_order_quantity_tons_rmse": quantity[split]["baseline_comparison"]["rmse"],
                    "baseline_order_quantity_tons_r2": quantity[split]["baseline_comparison"]["r2"],
                },
            )
        )

    if quantity_dummy:
        source_dummy = "models/metrics/summary/quantity_optimizer_baseline_comparison_latest__mixed_context.json"
        expected_pairs = {
            (model, split)
            for model in ["DummyRegressor", "Ridge"]
            for split in ["train", "validation", "test"]
        }
        actual_pairs = {
            (row.get("model"), row.get("split"))
            for row in quantity_dummy.get("metrics", [])
        }
        _add_check(
            checks,
            "quantity optimizer supervised comparison contains DummyRegressor and Ridge on all splits",
            actual_pairs == expected_pairs,
            f"pairs={sorted(actual_pairs)}",
        )
        feature_columns = set(quantity_dummy.get("feature_columns", []))
        _add_check(
            checks,
            "quantity optimizer supervised comparison excludes target and trigger label from features",
            "quantity_optimizer_target_tons" not in feature_columns and "purchase_trigger_label" not in feature_columns,
            f"feature_count={len(feature_columns)}",
        )
        dummy_mean = float(pd.to_numeric(prediction_frames["train"]["actual"], errors="coerce").mean())
        for row in quantity_dummy.get("metrics", []):
            split = row["split"]
            model = row["model"]
            predictions = prediction_frames[split]
            if model == "DummyRegressor":
                y_pred = pd.Series(dummy_mean, index=predictions.index)
            else:
                y_pred = predictions["prediction"]
            recalculated = compute_regression_metrics(
                predictions["actual"],
                y_pred,
                split=split,
                model=model,
                target=quantity_dummy["target"],
                include_mape=False,
                include_distribution=False,
            )
            expected = {
                "rmse": row["rmse"],
                "mae": row["mae"],
                "r2": row["r2"],
                "n_rows": row["n_rows"],
            }
            actual = {
                "rmse": recalculated["rmse"],
                "mae": recalculated["mae"],
                "r2": recalculated["r2"],
                "n_rows": recalculated["n_samples"],
            }
            _compare_fields(
                checks,
                name=f"quantity optimizer supervised {model} {split}: recalculated vs comparison JSON",
                expected=expected,
                actual=actual,
                fields=["rmse", "mae", "r2", "n_rows"],
            )
            dummy_rows.append(
                _metric_record(
                    section="quantity_optimizer_baseline_comparison",
                    source=source_dummy,
                    values=dict(row),
                )
            )
    return rows, dummy_rows


def _collect_policy(
    checks: list[Check],
    config: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    period_df = _read_relative_csv(policy["period_metrics_csv_path"])
    allowed = float(config["policy_simulation"]["kpi_definition"]["allowed_stockout_increase_pct"])
    recalculated = compute_policy_metrics(
        period_df,
        guardrail_name="bounded_stockout_increase",
        allowed_stockout_increase_pct=allowed,
    )
    _compare_fields(
        checks,
        name="policy simulation: period CSV vs summary JSON",
        expected=policy,
        actual=recalculated,
        fields=[
            "n_periods",
            "baseline_excess_tons",
            "policy_excess_tons",
            "absolute_excess_reduction_tons",
            "aggregate_excess_reduction_pct",
            "baseline_stockout_tons",
            "policy_stockout_tons",
            "aggregate_stockout_change_pct",
            "stockout_guardrail_pass",
        ],
    )
    return _metric_record(
        section="policy_simulation",
        source=(
            "models/metrics/summary/policy_simulation_latest__mixed_context.json; "
            f"{policy['period_metrics_csv_path']}"
        ),
        values={
            "n_periods": policy["n_periods"],
            "baseline_excess_tons": policy["baseline_excess_tons"],
            "policy_excess_tons": policy["policy_excess_tons"],
            "absolute_excess_reduction_tons": policy["absolute_excess_reduction_tons"],
            "aggregate_excess_reduction_pct": policy["aggregate_excess_reduction_pct"],
            "baseline_stockout_tons": policy["baseline_stockout_tons"],
            "policy_stockout_tons": policy["policy_stockout_tons"],
            "stockout_tons": policy["stockout_tons"],
            "aggregate_stockout_change_pct": policy["aggregate_stockout_change_pct"],
            "stockout_guardrail_pass": policy["stockout_guardrail_pass"],
            "trigger_rule_respected": policy["trigger_rule_respected"],
            "baseline_policy_name": policy["baseline_policy_name"],
            "proposed_policy_name": policy["proposed_policy_name"],
            "allowed_stockout_increase_pct": allowed,
        },
    )


def _collect_formula(
    checks: list[Check],
    config: dict[str, Any],
    formula_report: dict[str, Any],
) -> dict[str, Any]:
    configured = config["synthetic_data"]["simulation_parameters"]["procurement_target_parameters"]
    effective = _merge_procurement_target_parameters(config["synthetic_data"]["simulation_parameters"])
    fields = [
        "canonical_variant",
        "pressure_snapshot_blend_weight",
        "pressure_coverage_gap_blend_weight",
    ]
    for field in fields:
        _add_check(
            checks,
            f"formula {field}: config vs effective code parameters",
            _same_number(configured[field], effective[field]),
            f"config={configured[field]!r} effective={effective[field]!r}",
        )
        _add_check(
            checks,
            f"formula {field}: config vs generated formula report",
            _same_number(configured[field], formula_report[field]),
            f"config={configured[field]!r} report={formula_report[field]!r}",
        )
    pressure_snapshot = float(effective["pressure_snapshot_blend_weight"])
    pressure_coverage = float(effective["pressure_coverage_gap_blend_weight"])
    pressure_core = max(1.0 - pressure_snapshot - pressure_coverage, 0.0)
    return _metric_record(
        section="synthetic_procurement_need_formula",
        source="config/config.yaml; src/data_processing/synthetic_plant.py; reports/official/synthetic_procurement_need_formula__mixed_context.json",
        values={
            "canonical_variant": effective["canonical_variant"],
            "canonical_target_column": "synthetic_procurement_need",
            "effective_variant_column": f"synthetic_procurement_need_{effective['canonical_variant']}",
            "pressure_core_weight": pressure_core,
            "pressure_snapshot_blend_weight": pressure_snapshot,
            "pressure_coverage_gap_blend_weight": pressure_coverage,
            "forward_requirement_weights": [float(value) for value in effective["forward_requirement_weights"]],
            "formula_note": (
                "For canonical_variant=pressure, synthetic_procurement_need is the pressure variant: "
                "pressure_core_weight * pressure_core + pressure_snapshot_blend_weight * legacy_snapshot_need + "
                "pressure_coverage_gap_blend_weight * synthetic_procurement_need_coverage_gap."
            ),
        },
    )


def _iter_scan_files() -> list[Path]:
    files: list[Path] = []
    for root in DOC_SCAN_PATHS:
        if root.is_file():
            files.append(root)
            continue
        if root.exists():
            files.extend(
                path
                for path in sorted(root.rglob("*"))
                if path.is_file() and path.suffix.lower() in {".md", ".json"}
            )
    return files


def _scan_docs_for_obsolete_values(checks: list[Check]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for path in _iter_scan_files():
        text = path.read_text(encoding="utf-8")
        for label, pattern in OBSOLETE_PATTERNS.items():
            if pattern.search(text):
                findings.append({"path": _relative(path), "pattern": label})
    _add_check(
        checks,
        "documentation scan: no known stale metrics in README/docs/reports/official",
        not findings,
        "none" if not findings else json.dumps(findings, ensure_ascii=False),
    )
    return findings


def _load_inputs(config_path: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = build_reproducibility_config(config_path, mode=SMOKE_MODE)
    paths = official_paths(config)
    required_keys = [
        "baseline_summary_json",
        "trigger_metrics_json",
        "quantity_optimizer_metrics_json",
        "policy_simulation_summary_json",
        "official_formula_json",
        "metrics_summary_json",
    ]
    missing = [str(paths[key]) for key in required_keys if not paths[key].exists()]
    if paths.get("quantity_optimizer_baseline_comparison_json") and not paths[
        "quantity_optimizer_baseline_comparison_json"
    ].exists():
        missing.append(str(paths["quantity_optimizer_baseline_comparison_json"]))
    if missing:
        raise FileNotFoundError(f"Missing official metric artifacts: {missing}")
    payloads = {
        "baseline": read_json(paths["baseline_summary_json"]),
        "trigger": read_json(paths["trigger_metrics_json"]),
        "quantity": read_json(paths["quantity_optimizer_metrics_json"]),
        "quantity_baseline_comparison": read_json(paths["quantity_optimizer_baseline_comparison_json"])
        if paths["quantity_optimizer_baseline_comparison_json"].exists()
        else {},
        "policy": read_json(paths["policy_simulation_summary_json"]),
        "formula": read_json(paths["official_formula_json"]),
        "metrics_summary": read_json(paths["metrics_summary_json"]),
    }
    return config, paths, payloads


def build_alignment_payload(config_path: str = "config/config.yaml") -> dict[str, Any]:
    config, paths, payloads = _load_inputs(config_path)
    checks: list[Check] = []
    upstream = _collect_upstream(checks, payloads["baseline"])
    trigger = _collect_trigger(checks, payloads["trigger"])
    quantity, quantity_baseline = _collect_quantity(
        checks,
        payloads["quantity"],
        payloads["quantity_baseline_comparison"],
    )
    policy = _collect_policy(checks, config, payloads["policy"])
    formula = _collect_formula(checks, config, payloads["formula"])
    stale_findings = _scan_docs_for_obsolete_values(checks)

    summary_qo_comparison = payloads["metrics_summary"].get("quantity_optimizer_baseline_comparison", {})
    _add_check(
        checks,
        "metrics summary exposes quantity optimizer DummyRegressor comparison",
        bool(summary_qo_comparison)
        and summary_qo_comparison.get("target") == payloads["quantity_baseline_comparison"].get("target"),
        f"target={summary_qo_comparison.get('target')!r}",
    )

    passed = sum(check.passed for check in checks)
    return {
        "schema_version": 1,
        "scope": SCOPE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_date": config["official_release"]["reference_date"],
        "status": "PASS" if passed == len(checks) else "FAIL",
        "checks_passed": passed,
        "checks_total": len(checks),
        "checks": [asdict(check) for check in checks],
        "sources": {
            "baseline_summary_json": _relative(paths["baseline_summary_json"]),
            "trigger_metrics_json": _relative(paths["trigger_metrics_json"]),
            "quantity_optimizer_metrics_json": _relative(paths["quantity_optimizer_metrics_json"]),
            "quantity_optimizer_baseline_comparison_json": _relative(
                paths["quantity_optimizer_baseline_comparison_json"]
            ),
            "policy_simulation_summary_json": _relative(paths["policy_simulation_summary_json"]),
            "official_formula_json": _relative(paths["official_formula_json"]),
            "metrics_summary_json": _relative(paths["metrics_summary_json"]),
            "config": "config/config.yaml",
        },
        "documentable_metrics": {
            "upstream_predictor": upstream,
            "purchase_trigger": trigger,
            "quantity_optimizer": quantity,
            "quantity_optimizer_baseline_comparison": quantity_baseline,
            "policy_simulation": policy,
            "synthetic_procurement_need_formula": formula,
        },
        "documentation_scan": {
            "paths_scanned": [_relative(path) for path in _iter_scan_files()],
            "obsolete_findings": stale_findings,
        },
    }


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return repr(value)
    return str(value)


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# CU28 doc-code metrics alignment",
        "",
        f"- Scope: `{payload['scope']}`",
        f"- Reference date: `{payload['reference_date']}`",
        f"- Status: `{payload['status']}`",
        f"- Checks: `{payload['checks_passed']}/{payload['checks_total']}`",
        "",
        "## Checks",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for check in payload["checks"]:
        lines.append(
            f"| {check['name']} | {'PASS' if check['passed'] else 'FAIL'} | {check['detail'].replace('|', '/')} |"
        )

    lines.extend(
        [
            "",
            "## Upstream Predictor",
            "",
            "| Model | Split | n | MAE | RMSE | R2 | MAPE | Source |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["documentable_metrics"]["upstream_predictor"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model_family"],
                    row["split"],
                    _format_value(row["n_samples"]),
                    _format_value(row["mae"]),
                    _format_value(row["rmse"]),
                    _format_value(row["r2"]),
                    _format_value(row["mape"]),
                    row["source"],
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Purchase Trigger",
            "",
            "| Split | n | BUY pct | Accuracy | Balanced accuracy | Recall BUY | FNR BUY | Recall DO_NOT_BUY | Precision DO_NOT_BUY | F1 DO_NOT_BUY | TN | FP | FN | TP |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["documentable_metrics"]["purchase_trigger"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["split"],
                    _format_value(row["n_samples"]),
                    _format_value(row["buy_pct"]),
                    _format_value(row["accuracy"]),
                    _format_value(row["balanced_accuracy"]),
                    _format_value(row["recall_buy"]),
                    _format_value(row["false_negative_rate"]),
                    _format_value(row["recall_do_not_buy"]),
                    _format_value(row["precision_do_not_buy"]),
                    _format_value(row["f1_do_not_buy"]),
                    _format_value(row["true_negative"]),
                    _format_value(row["false_positive"]),
                    _format_value(row["false_negative"]),
                    _format_value(row["true_positive"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Quantity Optimizer",
            "",
            "| Model | Split | n | MAE | RMSE | R2 | Baseline MAE | Baseline RMSE | Baseline R2 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["documentable_metrics"]["quantity_optimizer"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    row["split"],
                    _format_value(row["n_samples"]),
                    _format_value(row["mae"]),
                    _format_value(row["rmse"]),
                    _format_value(row["r2"]),
                    _format_value(row["baseline_order_quantity_tons_mae"]),
                    _format_value(row["baseline_order_quantity_tons_rmse"]),
                    _format_value(row["baseline_order_quantity_tons_r2"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Quantity Optimizer Dummy Comparison",
            "",
            "| Model | Split | RMSE | MAE | R2 | n_rows |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["documentable_metrics"]["quantity_optimizer_baseline_comparison"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    row["split"],
                    _format_value(row["rmse"]),
                    _format_value(row["mae"]),
                    _format_value(row["r2"]),
                    _format_value(row["n_rows"]),
                ]
            )
            + " |"
        )

    policy = payload["documentable_metrics"]["policy_simulation"]
    lines.extend(
        [
            "",
            "## Policy Simulation",
            "",
            f"- `baseline_excess_tons`: `{_format_value(policy['baseline_excess_tons'])}`",
            f"- `policy_excess_tons`: `{_format_value(policy['policy_excess_tons'])}`",
            f"- `absolute_excess_reduction_tons`: `{_format_value(policy['absolute_excess_reduction_tons'])}`",
            f"- `aggregate_excess_reduction_pct`: `{_format_value(policy['aggregate_excess_reduction_pct'])}`",
            f"- `baseline_stockout_tons`: `{_format_value(policy['baseline_stockout_tons'])}`",
            f"- `policy_stockout_tons`: `{_format_value(policy['policy_stockout_tons'])}`",
            f"- `aggregate_stockout_change_pct`: `{_format_value(policy['aggregate_stockout_change_pct'])}`",
            f"- `stockout_guardrail_pass`: `{_format_value(policy['stockout_guardrail_pass'])}`",
            "",
            "## synthetic_procurement_need Formula",
            "",
        ]
    )
    formula = payload["documentable_metrics"]["synthetic_procurement_need_formula"]
    for key in [
        "canonical_variant",
        "effective_variant_column",
        "pressure_core_weight",
        "pressure_snapshot_blend_weight",
        "pressure_coverage_gap_blend_weight",
        "forward_requirement_weights",
        "formula_note",
    ]:
        lines.append(f"- `{key}`: `{_format_value(formula[key])}`")

    lines.extend(
        [
            "",
            "## Sources",
            "",
        ]
    )
    for key, source in payload["sources"].items():
        lines.append(f"- `{key}`: `{source}`")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit CU28 documentable metrics against official artifacts.")
    parser.add_argument("--scope", default=SCOPE)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--json-output", default=str(REPORT_JSON))
    parser.add_argument("--md-output", default=str(REPORT_MD))
    parser.add_argument("--fail-on-mismatch", action="store_true")
    args = parser.parse_args(argv)
    if args.scope != SCOPE:
        raise ValueError("Only scope=mixed_context is supported.")

    payload = build_alignment_payload(args.config)
    json_path = Path(args.json_output)
    md_path = Path(args.md_output)
    write_json(json_path, payload)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(payload), encoding="utf-8")

    failed = [check for check in payload["checks"] if not check["passed"]]
    print(
        json.dumps(
            {
                "scope": args.scope,
                "status": payload["status"],
                "checks": payload["checks_total"],
                "passed": payload["checks_passed"],
                "failed": len(failed),
                "json_report": str(json_path),
                "markdown_report": str(md_path),
                "failures": failed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if failed and args.fail_on_mismatch else 0


if __name__ == "__main__":
    raise SystemExit(main())
